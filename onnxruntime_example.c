#include <stdio.h>
#undef _WIN32
#include "onnxruntime_c_api.h"
#define _WIN32
#include <Windows.h>

typedef const OrtApiBase* (*GetOrtApiBaseFunction)(void);

// A global pointer to the OrtApi.
const OrtApi *ort_api = NULL;

#define CheckORTError(val) (InternalORTErrorCheck((val), #val, __FILE__, __LINE__))

static void InternalORTErrorCheck(OrtStatus *status, const char *text,
  const char *file, int line) {
  if (!status) return;
  printf("Got onnxruntime error %s, (%s at line %d in %s)\n",
    ort_api->GetErrorMessage(status), text, line, file);
  ort_api->ReleaseStatus(status);
  exit(1);
}

int main(int argc, char **argv) {
  HMODULE onnx_dll = NULL;
  OrtEnv *ort_env = NULL;
  GetOrtApiBaseFunction get_api_base_fn = NULL;
  const OrtApiBase *api_base = NULL;

  // Load the library and look up the function
  onnx_dll = LoadLibraryA("onnxruntime\\lib\\onnxruntime.dll");
  if (!onnx_dll) {
    printf("Failed loading onnxruntime.dll.\n");
    return 1;
  }
  get_api_base_fn = (GetOrtApiBaseFunction) GetProcAddress(onnx_dll,
    "OrtGetApiBase");
  if (get_api_base_fn == NULL) {
    printf("Couldn't find the Get API base function.\n");
    return 1;
  }
  printf("Get API base function @ %p\n", get_api_base_fn);

  // Actually get the API struct
  api_base = get_api_base_fn();
  if (!api_base) {
    printf("Failed getting API base.\n");
    return 1;
  }
  ort_api = api_base->GetApi(ORT_API_VERSION);
  if (!ort_api) {
    printf("Failed getting the ORT API.\n");
    return 1;
  }
  printf("ORT API @ %p\n", ort_api);

  // Create the environment.
  CheckORTError(ort_api->CreateEnv(ORT_LOGGING_LEVEL_VERBOSE, "Example", &ort_env));
  printf("Got ORT env @ %p\n", ort_env);
  ort_api->ReleaseEnv(ort_env);
  ort_env = NULL;
  printf("Released env OK.\n");
  return 0;
}
