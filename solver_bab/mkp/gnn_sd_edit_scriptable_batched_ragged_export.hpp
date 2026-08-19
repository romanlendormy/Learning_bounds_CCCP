#pragma once
/*
 * gnn_sd_edit_scriptable_batched_ragged_export.hpp — Export TorchScript
 * one-shot (fork python, mis en cache) du module RAGGED
 * (gnn_sd_edit/gnn_sd_edit_scriptable_batched_ragged.py), même mécanisme que
 * ensure_scripted_batched_export() (gnn_sd_edit_scriptable_batched_export.hpp)
 * mais suffixe de cache distinct (.ragged_scripted_batched.pt) — ne touche
 * jamais aux artefacts existants (.scripted.pt, .scripted_batched.pt).
 */

#include <cstring>
#include <spawn.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <vector>

extern char** environ;

namespace gnn_batched_export_ragged_detail {

inline bool file_exists(const std::string& p) { struct stat st; return ::stat(p.c_str(), &st) == 0; }
inline double file_mtime(const std::string& p) {
    struct stat st; if (::stat(p.c_str(), &st) != 0) return -1.0; return static_cast<double>(st.st_mtime);
}

} // namespace gnn_batched_export_ragged_detail

inline std::string ensure_scripted_batched_ragged_export(const std::string& checkpoint_path, const std::string& python_exe)
{
    using namespace gnn_batched_export_ragged_detail;
    std::string scripted_path = checkpoint_path + ".ragged_scripted_batched.pt";
    if (file_exists(scripted_path) && file_mtime(scripted_path) >= file_mtime(checkpoint_path))
        return scripted_path;

    std::string script = "gnn_sd_edit/gnn_sd_edit_scriptable_batched_ragged.py";

    std::vector<char*> argv = {
        const_cast<char*>(python_exe.c_str()), const_cast<char*>(script.c_str()),
        const_cast<char*>("--checkpoint"), const_cast<char*>(checkpoint_path.c_str()),
        const_cast<char*>("--out"), const_cast<char*>(scripted_path.c_str()),
        const_cast<char*>("--device"), const_cast<char*>("cpu"),
        nullptr
    };
    pid_t pid = 0;
    int rc = ::posix_spawnp(&pid, python_exe.c_str(), nullptr, nullptr, argv.data(), environ);
    if (rc != 0)
        throw std::runtime_error("ensure_scripted_batched_ragged_export: posix_spawnp a échoué: " + std::string(std::strerror(rc)));
    int status = 0;
    ::waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
        throw std::runtime_error("ensure_scripted_batched_ragged_export: gnn_sd_edit_scriptable_batched_ragged.py a échoué (checkpoint invalide ?)");
    return scripted_path;
}
