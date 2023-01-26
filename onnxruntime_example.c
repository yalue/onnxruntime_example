#include <stdio.h>
#include <stdlib.h>
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

// Compares the expected to the actual outputs produced by the NN. Returns 0 if
// anything doesn't match.
static int ValidateOutputs(float *expected, float *got, int length) {
  int i;
  float difference;
  for (i = 0; i < length; i++) {
    difference = got[i] - expected[i];
    if (difference < 0) difference = -difference;
    // The stuff we copy and paste from the python script isn't very precise!
    if (difference > 0.0001) {
      printf("Output mismatch detected; expected %f, got %f.\n", expected[i],
        got[i]);
      return 0;
    }
  }
  return 1;
}

int main(int argc, char **argv) {
  const char *model_path = "example_network.onnx";
  HMODULE onnx_dll = NULL;
  const OrtApiBase *api_base = NULL;
  GetOrtApiBaseFunction get_api_base_fn = NULL;
  OrtEnv *ort_env = NULL;
  OrtSessionOptions *options = NULL;
  OrtSession *session = NULL;
  OrtMemoryInfo *memory_info = NULL;
  OrtValue *input_tensor = NULL;
  OrtValue *output_tensor = NULL;
  OrtTensorTypeAndShapeInfo *output_info = NULL;

  // These were copied from the output of generate_network.py; update these
  // values if the network is ever re-generated.
  float input_data[] = {0.4088, 0.5113, 0.8682, 0.7237};
  int64_t input_shape[] = {1, 1, 4};
  const char *input_names[] = {"1x4 Input Vector"};
  float expected_output[] = {2.5120, 0.6187};
  const char *output_names[] = {"1x2 Output Vector"};
  float *output_values = NULL;
  size_t output_element_count = 0;

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
  CheckORTError(ort_api->CreateEnv(ORT_LOGGING_LEVEL_VERBOSE, "Example",
    &ort_env));


  // Create the session and load the model.
  printf("About to load %s\n", model_path);
  CheckORTError(ort_api->CreateSessionOptions(&options));
  // CreateSession expects a wide character string on Windows.
  CheckORTError(ort_api->CreateSession(ort_env,
    (char *) L"example_network.onnx", options, &session));
  printf("Loaded %s OK.\n", model_path);

  // Load the input data
  CheckORTError(ort_api->CreateCpuMemoryInfo(OrtArenaAllocator,
    OrtMemTypeDefault, &memory_info));
  CheckORTError(ort_api->CreateTensorWithDataAsOrtValue(memory_info,
    input_data, sizeof(input_data), input_shape, 3,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));

  // Actually run the inference
  CheckORTError(ort_api->Run(session, NULL, input_names,
    (const OrtValue* const*) &input_tensor, 1, output_names, 1,
    &output_tensor));

  // Get the output data from its tensor.
  CheckORTError(ort_api->GetTensorTypeAndShape(output_tensor, &output_info));
  CheckORTError(ort_api->GetTensorShapeElementCount(output_info,
    &output_element_count));
  if (output_element_count != 2) {
    printf("Expected to get 2 output values, got %d instead!\n",
      (int) output_element_count);
    exit(1);
  }
  CheckORTError(ort_api->GetTensorMutableData(output_tensor,
    (void **) (&output_values)));

  if (!ValidateOutputs(expected_output, output_values, 2)) {
    printf("WARNING: The network produced incorrect results!\n");
  } else {
    printf("The network produced the expected results.\n");
  }

  ort_api->ReleaseTensorTypeAndShapeInfo(output_info);
  ort_api->ReleaseValue(output_tensor);
  ort_api->ReleaseValue(input_tensor);
  ort_api->ReleaseMemoryInfo(memory_info);
  ort_api->ReleaseSession(session);
  ort_api->ReleaseSessionOptions(options);
  ort_api->ReleaseEnv(ort_env);
  ort_env = NULL;
  printf("Cleanup complete.\n");
  return 0;
}

