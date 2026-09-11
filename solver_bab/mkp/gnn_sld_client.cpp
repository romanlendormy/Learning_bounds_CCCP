#include "gnn_sld_client.hpp"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

// Note protocole : les messages sont des buffers d'octets à offsets explicites
// (jamais des struct C++ écrites telles quelles) — un struct { int32_t; ...;
// double; } insérerait un padding d'alignement que le protocole binaire de
// gnn_inference_server.py (struct.pack Python, format "<...", sans padding)
// n'a pas. memcpy à des offsets choisis à la main évite ce piège.

namespace {

void write_all_fd(int fd, const void* buf, size_t len)
{
    const char* p = static_cast<const char*>(buf);
    size_t remaining = len;
    while (remaining > 0) {
        ssize_t written = ::write(fd, p, remaining);
        if (written < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(std::string("gnn_sld_client: write() a échoué: ") + std::strerror(errno));
        }
        p += written;
        remaining -= static_cast<size_t>(written);
    }
}

// Retourne false sur EOF immédiat (aucun octet lu avant EOF) ; lève une
// exception si l'EOF survient au milieu d'un message (protocole rompu).
bool read_all_fd(int fd, void* buf, size_t len)
{
    char* p = static_cast<char*>(buf);
    size_t remaining = len;
    while (remaining > 0) {
        ssize_t n = ::read(fd, p, remaining);
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(std::string("gnn_sld_client: read() a échoué: ") + std::strerror(errno));
        }
        if (n == 0) {
            if (remaining == len) return false;
            throw std::runtime_error("gnn_sld_client: EOF inattendu du serveur GNN (protocole rompu)");
        }
        p += n;
        remaining -= static_cast<size_t>(n);
    }
    return true;
}

} // namespace

void GNNSLDClient::write_all(const void* buf, size_t len) { write_all_fd(fd_to_child_, buf, len); }

void GNNSLDClient::read_all(void* buf, size_t len)
{
    if (!read_all_fd(fd_from_child_, buf, len))
        throw std::runtime_error("gnn_sld_client: le serveur GNN a fermé la connexion de façon inattendue");
}

GNNSLDClient::GNNSLDClient(const std::string& python_exe,
                             const std::string& server_script,
                             const std::string& checkpoint_path,
                             const std::string& device,
                             int n, int m, int N, double epsilon,
                             const Vec1d& c, const Vec1d& p,
                             const Vec3d& A, const Vec2d& b,
                             const std::string& variant)
    : n_(n)
{
    // lagrangian_sld() (Python) suppose p_i = 1/N — refusé explicitement ici
    // plutôt que de laisser le serveur produire silencieusement une borne
    // invalide sur une instance à probabilités non-uniformes.
    const double p_unif = 1.0 / static_cast<double>(N);
    for (double pi : p)
        if (std::abs(pi - p_unif) > 1e-9)
            throw std::invalid_argument(
                "GNNSLDClient : p non-uniforme — lagrangian_sld suppose des "
                "scénarios équiprobables (p_i = 1/N)");

    int in_pipe[2];   // C++ écrit -> stdin de Python
    int out_pipe[2];  // Python écrit (stdout) -> C++ lit
    if (::pipe(in_pipe) != 0 || ::pipe(out_pipe) != 0)
        throw std::runtime_error("GNNSLDClient : pipe() a échoué");

    pid_t pid = ::fork();
    if (pid < 0)
        throw std::runtime_error("GNNSLDClient : fork() a échoué");

    if (pid == 0) {
        // ── Enfant : serveur Python ─────────────────────────────────────────
        ::dup2(in_pipe[0], STDIN_FILENO);
        ::dup2(out_pipe[1], STDOUT_FILENO);
        ::close(in_pipe[0]);  ::close(in_pipe[1]);
        ::close(out_pipe[0]); ::close(out_pipe[1]);

        ::execlp(python_exe.c_str(), python_exe.c_str(), server_script.c_str(),
                 "--checkpoint", checkpoint_path.c_str(),
                 "--device", device.c_str(),
                 "--variant", variant.c_str(),
                 static_cast<char*>(nullptr));
        // execlp ne revient qu'en cas d'échec du spawn.
        std::fprintf(stderr, "GNNSLDClient : execlp(%s) a échoué : %s\n",
                     python_exe.c_str(), std::strerror(errno));
        _exit(127);
    }

    // ── Parent : B&B C++ ─────────────────────────────────────────────────────
    child_pid_ = pid;
    ::close(in_pipe[0]);
    ::close(out_pipe[1]);
    fd_to_child_   = in_pipe[1];
    fd_from_child_ = out_pipe[0];

    // ── INIT : instance complète, envoyée une seule fois (offsets manuels,
    //    pas de struct, cf. note en tête de fichier) ──────────────────────────
    char header[3 * sizeof(int32_t) + sizeof(double)];
    int32_t n32 = n, m32 = m, N32 = N;
    std::memcpy(header,                                &n32,    sizeof(int32_t));
    std::memcpy(header + sizeof(int32_t),               &m32,    sizeof(int32_t));
    std::memcpy(header + 2 * sizeof(int32_t),            &N32,    sizeof(int32_t));
    std::memcpy(header + 3 * sizeof(int32_t),            &epsilon, sizeof(double));
    write_all(header, sizeof(header));

    std::vector<double> buf;
    buf.assign(c.begin(), c.end());
    write_all(buf.data(), buf.size() * sizeof(double));

    buf.assign(p.begin(), p.end());
    write_all(buf.data(), buf.size() * sizeof(double));

    buf.clear(); buf.reserve(static_cast<size_t>(N) * m * n);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < m; ++j)
            for (int k = 0; k < n; ++k)
                buf.push_back(A[i][j][k]);
    write_all(buf.data(), buf.size() * sizeof(double));

    buf.clear(); buf.reserve(static_cast<size_t>(N) * m);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < m; ++j)
            buf.push_back(b[i][j]);
    write_all(buf.data(), buf.size() * sizeof(double));

    // ── ACK ─────────────────────────────────────────────────────────────────
    int32_t status = 0, msg_len = 0;
    read_all(&status, sizeof(status));
    read_all(&msg_len, sizeof(msg_len));
    std::string msg;
    if (msg_len > 0) {
        msg.resize(static_cast<size_t>(msg_len));
        read_all(msg.data(), msg.size());
    }
    if (status != 0) {
        int st = 0; ::waitpid(child_pid_, &st, 0);
        throw std::runtime_error("GNNSLDClient : le serveur GNN a rejeté l'instance : " + msg);
    }
}

GNNSLDClient::~GNNSLDClient()
{
    // Ordre important : fermer fd_from_child_ AVANT que Python ait fini de
    // sortir (après l'EOF déclenché par la fermeture de fd_to_child_) lui
    // ferait recevoir EPIPE sur son dernier flush de stdout (bruit inoffensif
    // mais évitable) — on attend la fin réelle du process avant de fermer
    // notre bout de lecture.
    if (fd_to_child_ >= 0) ::close(fd_to_child_);   // EOF côté Python -> sortie propre
    if (child_pid_ > 0) {
        int status = 0;
        ::waitpid(child_pid_, &status, 0);
    }
    if (fd_from_child_ >= 0) ::close(fd_from_child_);
}

double GNNSLDClient::query(const std::vector<int8_t>& x_mask, const Vec2d& /*b_res*/,
                             double curr_cost, int depth, GNNCallTiming& timing)
{
    // b_res non transmis au serveur (cf. commentaire du .hpp) : le protocole
    // binaire n'envoie que x_mask, le serveur Python recalcule de son côté.
    std::vector<char> req(static_cast<size_t>(n_) + sizeof(double) + sizeof(int32_t));
    std::memcpy(req.data(), x_mask.data(), static_cast<size_t>(n_));
    std::memcpy(req.data() + n_, &curr_cost, sizeof(double));
    int32_t depth32 = depth;
    std::memcpy(req.data() + n_ + sizeof(double), &depth32, sizeof(int32_t));
    write_all(req.data(), req.size());

    // Offsets manuels (cf. note en tête de fichier) : status(i32) + 4 doubles.
    char resp[sizeof(int32_t) + 4 * sizeof(double)];
    read_all(resp, sizeof(resp));

    int32_t status;
    double ub_relative, t_graph, t_forward, t_bound;
    std::memcpy(&status,       resp,                                  sizeof(int32_t));
    std::memcpy(&ub_relative,  resp + sizeof(int32_t),                 sizeof(double));
    std::memcpy(&t_graph,      resp + sizeof(int32_t) + sizeof(double),     sizeof(double));
    std::memcpy(&t_forward,    resp + sizeof(int32_t) + 2 * sizeof(double), sizeof(double));
    std::memcpy(&t_bound,      resp + sizeof(int32_t) + 3 * sizeof(double), sizeof(double));

    if (status != 0)
        throw std::runtime_error("GNNSLDClient : le serveur GNN a renvoyé une erreur au nœud");

    timing.t_graph_build_s   = t_graph;
    timing.t_forward_s       = t_forward;
    timing.t_bound_compute_s = t_bound;
    return ub_relative;
}
