#include "error.hpp"
#include "predictor.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iosfwd>
#include <iostream>
#include <memory>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>
#include <iomanip>

#if 0
#define DEBUG_STMT std ::cout << __func__ << "  " << __LINE__ << "\n";
#else
#define DEBUG_STMT
#endif

using namespace torch;
using std::string;
using namespace torch::autograd::profiler;

extern Torch_IValue Torch_ConvertIValueToTorchIValue(torch::IValue value);

class Predictor {
 public:
  Predictor(const string &model_file, Torch_DeviceKind device, bool profiling_enabled);
  void Predict(Torch_TensorContext *cInputs, int inputLength);
  torch::jit::script::Module net_;
  torch::IValue output_;
  torch::DeviceType mode_{torch::kCPU};

  bool profile_enabled_{false};
  int64_t profile_start_;
};

Predictor::Predictor(const string &model_file, Torch_DeviceKind device, bool profiling_enabled) {
  // Load the network
  net_ = torch::jit::load(model_file);
  if (device == CUDA_DEVICE_KIND) mode_ = torch::kCUDA;

  if (mode_ == torch::kCUDA) {
    net_.to(at::kCUDA);
  }

  profile_enabled_ = profiling_enabled;

}

void Predictor::Predict(Torch_TensorContext *cInputs, int inputLength) {
  torch::NoGradGuard no_grad_guard;
  std::vector<torch::jit::IValue> inputs{};

  for (int ii = 0; ii < inputLength; ii++) {
    at::Tensor tensor = reinterpret_cast<Torch_Tensor *>(cInputs[ii])->tensor;

#ifdef DEBUG
    std::cout << "tensor dim = " << tensor.dim() << " size = ";
    for (auto sz : tensor.sizes()) {
      std::cout << sz << ", ";
    }
    std::cout << "\n";
#endif

    inputs.emplace_back(tensor);
  }

  if (profile_enabled_ == true) {
    enableProfilerLegacy(ProfilerConfig(ProfilerState::CPU, true, true));
    output_ = net_.forward(inputs);
    return;
  }
 
  output_ = net_.forward(inputs);
}

Torch_PredictorContext Torch_NewPredictor(const char *model_file, Torch_DeviceKind mode, bool profiling_enabled) {
  HANDLE_TH_ERRORS(Torch_GlobalError);
  const auto ctx = new Predictor(model_file, mode, profiling_enabled);
  return (Torch_PredictorContext)ctx;
  END_HANDLE_TH_ERRORS(Torch_GlobalError, (Torch_PredictorContext)0);
}

void Torch_PredictorRun(Torch_PredictorContext pred, Torch_TensorContext *cInputs, int inputLength) {
  HANDLE_TH_ERRORS(Torch_GlobalError);
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  predictor->Predict(cInputs, inputLength);
  END_HANDLE_TH_ERRORS(Torch_GlobalError, );
}

int Torch_PredictorNumOutputs(Torch_PredictorContext pred) {
  HANDLE_TH_ERRORS(Torch_GlobalError);
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return 0;
  }
  if (predictor->output_.isTensor()) {
    return 1;
  }
  if (predictor->output_.isTuple()) {
    return predictor->output_.toTuple()->elements().size();
  }

  return 0;
  END_HANDLE_TH_ERRORS(Torch_GlobalError, 0);
}

Torch_IValue Torch_PredictorGetOutput(Torch_PredictorContext pred) {
  HANDLE_TH_ERRORS(Torch_GlobalError);
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return Torch_IValue{};
  }

  return Torch_ConvertIValueToTorchIValue(predictor->output_);

  END_HANDLE_TH_ERRORS(Torch_GlobalError, Torch_IValue{});
}

void Torch_PredictorDelete(Torch_PredictorContext pred) {
  HANDLE_TH_ERRORS(Torch_GlobalError);
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }

  delete predictor;
  END_HANDLE_TH_ERRORS(Torch_GlobalError, );
}

char *Torch_ProfilingRead(Torch_PredictorContext pred) {
  HANDLE_TH_ERRORS(Torch_GlobalError);
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return strdup("");
  }

  auto event_lists = disableProfilerLegacy();
  std::vector < LegacyEvent * > events;
  for (auto & l: event_lists) {
    for (auto & e: l) {
      events.push_back( & e);
    }
  }

  std::stringstream out;

  auto time_correction = [](const int64_t& t) -> int64_t {
    auto t1 = getTime();
    auto t2 = static_cast<int64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
    auto t3 = getTime();
    return t + (t2 - (t1 + t3) / 2);
  };

  LegacyEvent * profiler_start = nullptr;
  for (LegacyEvent * e: events) {
    if (0 == strcmp(e -> name(), "__start_profile")) {
      predictor->profile_start_ = time_correction(static_cast<int64_t>(e -> cpuUs() * 1000));
      profiler_start = e;
      break;
    }
  }

  TORCH_CHECK(profiler_start, "Could not find __start_profile mark");

  struct PairHash {
    size_t operator()(std::pair < at::RecordFunctionHandle, int > p) const
      noexcept {
        return std::hash < at::RecordFunctionHandle > ()(p.first) ^ std::hash < int64_t > ()(p.second);
      }
  };

  std::unordered_map < std::pair < at::RecordFunctionHandle, int64_t > , LegacyEvent * , PairHash > events_map;
  bool first = true;
  std::vector < double > st;
  std::vector < double > dur;
  std::vector < std::string > name;
  std::vector < std::string > shapes;
  std::vector < int64_t > allocated_memory;
  std::vector < int64_t > peak_memory;
  std::vector < int64_t > tid;
  int64_t cur_mem = 0, cnt = 0;
  for (LegacyEvent * evt: events) {
    if (evt -> kindStr() == "push") {
      events_map[std::make_pair(evt -> handle(), evt -> nodeId())] = evt;
      if (0 == strcmp(evt -> name(), "forward")) {
        continue;
      }
      if (cnt == 0) {
        name.emplace_back(evt -> name());
        st.emplace_back(profiler_start -> cpuElapsedUs( * evt));
        std::string cur_shape = "[";
        const auto& sh = evt -> shapes();
        for (size_t i = 0; i < sh.size(); i++) {
          const auto & v = sh[i];
          if (i != 0) {
            cur_shape += ", ";
          }
          cur_shape += "[";
          for (size_t j = 0; j < v.size(); j++) {
            if (j != 0) {
              cur_shape += ", ";
            }
            cur_shape += std::to_string(v[j]);
          }
          cur_shape += "]";
        }
        cur_shape += "]";
        shapes.emplace_back(cur_shape);
        allocated_memory.emplace_back(0);
        peak_memory.emplace_back(0);
        cur_mem = 0;
      }
      cnt++;
    } else if (evt -> kindStr() == "pop") {
      auto it = events_map.find(std::make_pair(evt -> handle(), evt -> nodeId()));
      TORCH_CHECK(it != events_map.end(), "Unmatched pop event");
      LegacyEvent * evt_start = it -> second;
      events_map.erase(it);
      if (0 == strcmp(evt_start -> name(), "forward")) {
        continue;
      }
      cnt--;
      if(cnt == 0) {
      	dur.emplace_back(evt_start -> cpuElapsedUs( * evt));
      	tid.emplace_back(evt_start -> threadId());	
      }
    } else if (evt -> kindStr() == "memory_alloc") {
      if (cnt == 0) {
        continue;
      }
      cur_mem += evt -> cpuMemoryUsage();
      if (evt -> cpuMemoryUsage() > 0) {
        allocated_memory.back() += evt -> cpuMemoryUsage();
      }
      if (peak_memory.back() < cur_mem) {
        peak_memory.back() = cur_mem;
      }
    }
  }
  auto outputLong = [&](const std::string& key, const int64_t& val) -> void {
   	out << "  \"" << key << "\": " << val << ",\n";
  };
  auto outputFloat = [&](const std::string& key, const double& val) -> void {
   	out << "  \"" << key << "\": " << std::fixed << std::setprecision(3) << val << ",\n";
  };
  auto outputString = [&](const std::string& key, const std::string& val) -> void {
    out << "  \"" << key << "\": \"" << val << "\",\n";
  };
  out << "[\n";
  for (size_t i = 0; i < name.size(); i++) {
    if (i != 0) {
    	out << ",\n";
    }
    out << "{\n";
    outputString("name", name[i]);
    outputString("ph", "X");
    outputFloat("ts", st[i]);
    outputFloat("dur", dur[i]);
    outputLong("tid", tid[i]);
    outputString("pid", "CPU Functions");
    outputString("shape", shapes[i]);
    outputLong("allocated_memory", allocated_memory[i]);
    outputLong("peak_memory", peak_memory[i]);
    outputLong("layer_sequence_index", i);
    out << "  \"" << "args" << "\": " << "{}" << "\n";
    out << "}\n";
  }
  out << "]\n";

  return strdup(out.str().c_str());

  END_HANDLE_TH_ERRORS(Torch_GlobalError, (char *)0);
}

int64_t Torch_ProfilingGetStartTime(Torch_PredictorContext pred) {
  HANDLE_TH_ERRORS(Torch_GlobalError);
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return 0;
  }

  return predictor->profile_start_;
  END_HANDLE_TH_ERRORS(Torch_GlobalError, 0);
}

