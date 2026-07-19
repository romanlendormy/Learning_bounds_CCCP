// gnn_aoti_compat_probe.cpp — sonde de compatibilité de version, ÉTAPE A de
// l'investigation AOTInductor (accélération inférence GNN sd_edit).
//
// Charge un artefact AOTInductor (.so) produit par le venv Python
// (torch 2.12.1, cf. gnn_sd_edit/aoti_compat_probe.py) via le runtime C++
// figé sur libtorch 2.5.1 (external/libtorch). Teste si l'écart de version
// casse le chargement/l'exécution (question ouverte, jamais testée avant
// cette sonde) — modèle JOUET, pas le vrai GNN sd_edit.
//
// Usage : gnn_aoti_compat_probe <chemin_vers_le.so> [device=cpu|cuda]
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>
#ifdef GNN_AOTI_WITH_CUDA
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>
#endif
#include <torch/torch.h>

#include <iostream>
#include <memory>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <chemin_vers_le.so> [device=cpu|cuda]\n";
        return 1;
    }
    const std::string so_path = argv[1];
    const std::string device = (argc > 2) ? argv[2] : "cpu";

    try {
        std::cout << "[C++] chargement de : " << so_path << " (device=" << device << ")\n";

        std::unique_ptr<torch::inductor::AOTIModelContainerRunner> runner;
        auto x = torch::arange(8, torch::kFloat32).unsqueeze(0);
        if (device == "cuda") {
#ifdef GNN_AOTI_WITH_CUDA
            runner = std::make_unique<torch::inductor::AOTIModelContainerRunnerCuda>(so_path);
            x = x.to(torch::kCUDA);
#else
            std::cerr << "[C++] ECHEC : binaire compile sans support CUDA (GNN_AOTI_WITH_CUDA)\n";
            return 3;
#endif
        } else {
            runner = std::make_unique<torch::inductor::AOTIModelContainerRunnerCpu>(so_path);
        }

        std::vector<at::Tensor> inputs{x};
        auto outputs = runner->run(inputs);
        std::cout << "[C++] OK — sortie : " << outputs[0] << "\n";
        std::cout << "[C++] SUCCES : artefact AOTInductor charge par ce runtime libtorch\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[C++] ECHEC : " << e.what() << "\n";
        return 2;
    }
}
