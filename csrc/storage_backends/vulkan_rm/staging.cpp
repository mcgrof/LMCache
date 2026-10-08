// SPDX-License-Identifier: Apache-2.0
// RM declarations come from NVIDIA's MIT-licensed 615.71.09 SDK headers.
// ABI source:
// NVIDIA/open-gpu-kernel-modules@61dcc93722ecb418bb5f2e00923f05b4b8051dd1
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vulkan/vulkan.h>

#include <class/cl0040.h>
#include <class/cl0080.h>
#include <class/cl2080.h>
#include <ctrl/ctrl0000/ctrl0000client.h>
#include <ctrl/ctrl0000/ctrl0000gpu.h>
#include <ctrl/ctrl0000/ctrl0000unix.h>
#include <ctrl/ctrl0041.h>
#include <nv-ioctl.h>
#include <nv_escape.h>
#include <nvmisc.h>
#include <nvos.h>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {
constexpr char kVersion[] = "615.71.09";
constexpr uint64_t kMaxBytes = 1ULL << 30;
std::atomic<bool> quarantined{false};

void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error("Vulkan/RM staging: " + message);
}
void cuda_check(CUresult result, const char* stage) {
  require(result == CUDA_SUCCESS,
          std::string(stage) + " CUresult=" + std::to_string(result));
}
void vk_check(VkResult result, const char* stage) {
  require(result == VK_SUCCESS,
          std::string(stage) + " VkResult=" + std::to_string(result));
}
bool ioctl_ok(int fd, unsigned command, void* data, size_t bytes) {
  return ioctl(fd, _IOC(_IOC_READ | _IOC_WRITE, NV_IOCTL_MAGIC, command, bytes),
               data) == 0;
}
void rm_control(int fd, NvHandle client, NvHandle object, NvU32 command,
                void* data, size_t bytes) {
  NVOS54_PARAMETERS p{};
  p.hClient = client;
  p.hObject = object;
  p.cmd = command;
  p.params = reinterpret_cast<NvP64>(data);
  p.paramsSize = bytes;
  p.status = UINT32_MAX;
  bool ok = ioctl_ok(fd, NV_ESC_RM_CONTROL, &p, sizeof(p));
  require(ok && p.status == 0, "RM control " + std::to_string(command) +
                                   " errno=" + std::to_string(ok ? 0 : errno) +
                                   " status=" + std::to_string(p.status));
}
NvHandle rm_allocate(int fd, NvHandle client, NvHandle parent, NvHandle handle,
                     NvU32 class_id, void* data, size_t bytes) {
  NVOS21_PARAMETERS p{};
  p.hRoot = client;
  p.hObjectParent = parent;
  p.hObjectNew = handle;
  p.hClass = class_id;
  p.pAllocParms = reinterpret_cast<NvP64>(data);
  p.paramsSize = bytes;
  p.status = UINT32_MAX;
  bool ok = ioctl_ok(fd, NV_ESC_RM_ALLOC, &p, sizeof(p));
  require(ok && p.status == 0 && p.hObjectNew,
          "RM allocate errno=" + std::to_string(ok ? 0 : errno) +
              " status=" + std::to_string(p.status));
  return p.hObjectNew;
}
bool rm_free(int fd, NvHandle client, NvHandle parent, NvHandle object) {
  if (!object) return true;
  NVOS00_PARAMETERS p{};
  p.hRoot = client;
  p.hObjectParent = parent;
  p.hObjectOld = object;
  p.status = UINT32_MAX;
  return ioctl_ok(fd, NV_ESC_RM_FREE, &p, sizeof(p)) && p.status == 0;
}

// The tensor's storage deleter owns this graph, not a normal CUDA allocator.
// Registrations borrow a duplicate DMA-BUF FD while that storage remains live.
struct Allocation {
  VkInstance instance = VK_NULL_HANDLE;
  VkDevice vk_device = VK_NULL_HANDLE;
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  CUcontext context = nullptr;
  CUexternalMemory external = nullptr;
  CUdeviceptr pointer = 0;
  int opaque = -1, cuda_fd = -1, ctl = -1, gpu_fd = -1, dma_fd = -1;
  NvHandle client = 0, device = 0, subdevice = 0, rm_memory = 0;
  std::string identity;

  bool cleanup() noexcept {
    // No Vulkan queue is ever submitted. CUDA quiescence is necessary even
    // after the last tensor view dies: queued kernels may have dropped refs.
    if (context && (external || pointer)) {
      if (cuCtxPushCurrent(context) != CUDA_SUCCESS) return false;
      bool ok = cuCtxSynchronize() == CUDA_SUCCESS;
      if (ok && pointer) {
        ok = cuMemFree(pointer) == CUDA_SUCCESS;
        if (ok) pointer = 0;
      }
      if (ok && external) {
        ok = cuDestroyExternalMemory(external) == CUDA_SUCCESS;
        if (ok) external = nullptr;
      }
      CUcontext previous;
      if (cuCtxPopCurrent(&previous) != CUDA_SUCCESS) ok = false;
      if (!ok) return false;
    }
    // An error stops destruction of the remaining graph; never guess drain.
    for (int* fd : {&dma_fd, &cuda_fd, &opaque}) {
      if (*fd >= 0) {
        int value = *fd;
        *fd = -1;  // Linux close must not be retried on an uncertain result.
        if (close(value)) return false;
      }
    }
    if (!rm_free(ctl, client, device, rm_memory)) return false;
    rm_memory = 0;
    if (!rm_free(ctl, client, device, subdevice)) return false;
    subdevice = 0;
    if (!rm_free(ctl, client, client, device)) return false;
    device = 0;
    if (!rm_free(ctl, client, 0, client)) return false;
    client = 0;
    for (int* fd : {&gpu_fd, &ctl}) {
      if (*fd >= 0) {
        int value = *fd;
        *fd = -1;
        if (close(value)) return false;
      }
    }
    if (buffer) vkDestroyBuffer(vk_device, buffer, nullptr);
    if (memory) vkFreeMemory(vk_device, memory, nullptr);
    if (vk_device) vkDestroyDevice(vk_device, nullptr);
    if (instance) vkDestroyInstance(instance, nullptr);
    return true;
  }
};

void release(Allocation* allocation) noexcept {
  if (allocation->cleanup()) {
    delete allocation;
  } else {
    // Deliberate process-lifetime retention. Stop all further admissions;
    // process teardown, not a timeout or a destructor retry, reclaims it.
    quarantined.store(true);
    std::fprintf(stderr,
                 "Vulkan/RM staging: cleanup failed; owners retained, "
                 "provider admission sealed\n");
  }
}

void create_vulkan(Allocation& a, const CUuuid& uuid, uint64_t bytes,
                   VkDeviceSize& capacity) {
  VkApplicationInfo app{};
  app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app.pApplicationName = "lmcache-staging";
  app.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo instance_info{};
  instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instance_info.pApplicationInfo = &app;
  vk_check(vkCreateInstance(&instance_info, nullptr, &a.instance), "instance");
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  std::vector<VkPhysicalDevice> devices;
  bool complete = false;
  for (int attempt = 0; attempt < 8; ++attempt) {
    uint32_t count = 0;
    vk_check(vkEnumeratePhysicalDevices(a.instance, &count, nullptr),
             "devices");
    require(count && count <= 1024, "invalid Vulkan device count");
    devices.resize(count);
    auto result =
        vkEnumeratePhysicalDevices(a.instance, &count, devices.data());
    if (result == VK_INCOMPLETE) continue;
    vk_check(result, "device inventory");
    require(count && count <= devices.size(), "invalid Vulkan inventory");
    devices.resize(count);
    complete = true;
    break;
  }
  require(complete, "Vulkan inventory did not stabilize");
  for (auto candidate : devices) {
    VkPhysicalDeviceIDProperties id{};
    id.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &id;
    vkGetPhysicalDeviceProperties2(candidate, &properties);
    if (!std::memcmp(id.deviceUUID, uuid.bytes, 16)) {
      require(!physical, "ambiguous CUDA/Vulkan UUID");
      physical = candidate;
    }
  }
  require(physical, "CUDA UUID has no Vulkan physical device");
  constexpr auto handle = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
  constexpr auto usage =
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VkPhysicalDeviceExternalBufferInfo query{};
  query.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO;
  query.usage = usage;
  query.handleType = handle;
  VkExternalBufferProperties supported{};
  supported.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES;
  vkGetPhysicalDeviceExternalBufferProperties(physical, &query, &supported);
  const auto& external = supported.externalMemoryProperties;
  require((external.externalMemoryFeatures &
           VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) &&
              (external.compatibleHandleTypes & handle),
          "OPAQUE_FD buffer not exportable");
  uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
  require(count && count <= 4096, "invalid Vulkan queue count");
  std::vector<VkQueueFamilyProperties> queues(count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, queues.data());
  uint32_t family = 0;
  for (; family < count; ++family) {
    if (queues[family].queueCount &&
        (queues[family].queueFlags &
         (VK_QUEUE_TRANSFER_BIT | VK_QUEUE_COMPUTE_BIT |
          VK_QUEUE_GRAPHICS_BIT)))
      break;
  }
  require(family < count, "no Vulkan queue family");
  float priority = 1;
  VkDeviceQueueCreateInfo queue{};
  queue.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue.queueFamilyIndex = family;
  queue.queueCount = 1;
  queue.pQueuePriorities = &priority;
  const char* extension = VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME;
  VkDeviceCreateInfo device_info{};
  device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_info.queueCreateInfoCount = 1;
  device_info.pQueueCreateInfos = &queue;
  device_info.enabledExtensionCount = 1;
  device_info.ppEnabledExtensionNames = &extension;
  vk_check(vkCreateDevice(physical, &device_info, nullptr, &a.vk_device),
           "device");
  VkExternalMemoryBufferCreateInfo export_buffer{};
  export_buffer.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
  export_buffer.handleTypes = handle;
  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.pNext = &export_buffer;
  buffer_info.size = bytes;
  buffer_info.usage = usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  vk_check(vkCreateBuffer(a.vk_device, &buffer_info, nullptr, &a.buffer),
           "buffer");
  VkMemoryRequirements requirements;
  vkGetBufferMemoryRequirements(a.vk_device, a.buffer, &requirements);
  capacity = requirements.size;
  require(capacity >= bytes && capacity <= kMaxBytes &&
              capacity % sysconf(_SC_PAGESIZE) == 0,
          "Vulkan backing extent must fit the 1 GiB experimental limit");
  VkPhysicalDeviceMemoryProperties memory_properties;
  vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
  uint32_t type = 0;
  for (; type < memory_properties.memoryTypeCount; ++type) {
    auto flags = memory_properties.memoryTypes[type].propertyFlags;
    if ((requirements.memoryTypeBits & (1U << type)) &&
        (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
        !(flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
      break;
  }
  require(type < memory_properties.memoryTypeCount,
          "no device-local non-host-visible memory");
  VkMemoryDedicatedAllocateInfo dedicated{};
  dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
  dedicated.buffer = a.buffer;
  VkExportMemoryAllocateInfo export_info{};
  export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
  export_info.pNext = &dedicated;
  export_info.handleTypes = handle;
  VkMemoryAllocateInfo allocation_info{};
  allocation_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocation_info.pNext = &export_info;
  allocation_info.allocationSize = capacity;
  allocation_info.memoryTypeIndex = type;
  vk_check(vkAllocateMemory(a.vk_device, &allocation_info, nullptr, &a.memory),
           "memory");
  vk_check(vkBindBufferMemory(a.vk_device, a.buffer, a.memory, 0), "binding");
  auto get_fd = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
      vkGetDeviceProcAddr(a.vk_device, "vkGetMemoryFdKHR"));
  require(get_fd, "vkGetMemoryFdKHR unavailable");
  VkMemoryGetFdInfoKHR fd_info{};
  fd_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
  fd_info.memory = a.memory;
  fd_info.handleType = handle;
  vk_check(get_fd(a.vk_device, &fd_info, &a.opaque), "OPAQUE_FD export");
  require(a.opaque >= 0, "invalid OPAQUE_FD");
}

void export_rm(Allocation& a, const CUuuid& uuid, const char* bdf,
               uint64_t capacity, uint64_t bytes, uint64_t& backing_bytes) {
  a.ctl = open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
  require(a.ctl >= 0, "cannot open NVIDIA control device");
  nv_ioctl_rm_api_version_t version{};
  version.cmd = NV_RM_API_VERSION_CMD_STRICT;
  std::snprintf(version.versionString, sizeof(version.versionString), "%s",
                kVersion);
  require(
      ioctl_ok(a.ctl, NV_ESC_CHECK_VERSION_STR, &version, sizeof(version)) &&
          version.reply == NV_RM_API_VERSION_REPLY_RECOGNIZED,
      "unsupported NVIDIA RM ABI; requires 615.71.09");
  a.client = rm_allocate(a.ctl, 0, 0, 0, NV01_ROOT_CLIENT, nullptr, 0);
  NV0000_CTRL_GPU_GET_UUID_INFO_PARAMS id{};
  std::memcpy(id.gpuUuid, uuid.bytes, 16);
  id.flags = NV0000_CTRL_CMD_GPU_GET_UUID_INFO_FLAGS_FORMAT_BINARY;
  rm_control(a.ctl, a.client, a.client, NV0000_CTRL_CMD_GPU_GET_UUID_INFO, &id,
             sizeof(id));
  NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS gpu{};
  gpu.gpuId = id.gpuId;
  rm_control(a.ctl, a.client, a.client, NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2,
             &gpu, sizeof(gpu));
  require(gpu.deviceInstance == id.deviceInstance && !gpu.subDeviceInstance &&
              !id.subdeviceInstance &&
              DRF_VAL(0000, _CTRL_GPU_ID_INFO, _LINKED_INTO_SLI_DEVICE,
                      gpu.gpuFlags) !=
                  NV0000_CTRL_GPU_ID_INFO_LINKED_INTO_SLI_DEVICE_TRUE,
          "unsupported linked/subdevice identity");
  NV0000_CTRL_GPU_GET_UUID_FROM_GPU_ID_PARAMS reverse{};
  reverse.gpuId = gpu.gpuId;
  reverse.flags = NV0000_CTRL_CMD_GPU_GET_UUID_FROM_GPU_ID_FLAGS_FORMAT_BINARY;
  rm_control(a.ctl, a.client, a.client,
             NV0000_CTRL_CMD_GPU_GET_UUID_FROM_GPU_ID, &reverse,
             sizeof(reverse));
  require(!std::memcmp(reverse.gpuUuid, uuid.bytes, 16), "RM UUID mismatch");
  nv_ioctl_card_info_t cards[NV0000_CTRL_GPU_MAX_ATTACHED_GPUS]{};
  require(ioctl_ok(a.ctl, NV_ESC_CARD_INFO, cards, sizeof(cards)),
          "RM card inventory");
  unsigned domain, bus, slot, function;
  require(std::sscanf(bdf, "%x:%x:%x.%x", &domain, &bus, &slot, &function) == 4,
          "invalid CUDA BDF");
  int match = -1;
  for (unsigned i = 0; i < NV0000_CTRL_GPU_MAX_ATTACHED_GPUS; ++i) {
    if (cards[i].valid && cards[i].gpu_id == gpu.gpuId) {
      require(match == -1, "ambiguous RM GPU");
      match = i;
    }
  }
  require(match >= 0 && cards[match].pci_info.domain == domain &&
              cards[match].pci_info.bus == bus &&
              cards[match].pci_info.slot == slot &&
              cards[match].pci_info.function == function,
          "CUDA/RM BDF mismatch");
  NV0000_CTRL_OS_UNIX_GET_EXPORT_OBJECT_INFO_PARAMS info{};
  info.fd = a.opaque;
  rm_control(a.ctl, a.client, a.client,
             NV0000_CTRL_CMD_OS_UNIX_GET_EXPORT_OBJECT_INFO, &info,
             sizeof(info));
  require(info.maxObjects == 1 && info.deviceInstance == gpu.deviceInstance &&
              info.gpuInstanceId == UINT32_MAX,
          "foreign or unsupported Vulkan object");
  auto path = "/dev/nvidia" + std::to_string(cards[match].minor_number);
  a.gpu_fd = open(path.c_str(), O_RDWR | O_CLOEXEC);
  require(a.gpu_fd >= 0, "cannot open matched NVIDIA GPU");
  nv_ioctl_register_fd_t association{};
  association.ctl_fd = a.ctl;
  require(
      ioctl_ok(a.gpu_fd, NV_ESC_REGISTER_FD, &association, sizeof(association)),
      "RM FD association");
  NV0080_ALLOC_PARAMETERS device{};
  device.deviceId = gpu.deviceInstance;
  device.hClientShare = a.client;
  a.device = rm_allocate(a.ctl, a.client, a.client, 0x100, NV01_DEVICE_0,
                         &device, sizeof(device));
  NV2080_ALLOC_PARAMETERS subdevice{};
  a.subdevice = rm_allocate(a.ctl, a.client, a.device, 0x101, NV20_SUBDEVICE_0,
                            &subdevice, sizeof(subdevice));
  NV0000_CTRL_OS_UNIX_IMPORT_OBJECT_FROM_FD_PARAMS imported{};
  imported.fd = a.opaque;
  imported.object.type = NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TYPE_RM;
  imported.object.data.rmObject.hDevice = a.device;
  imported.object.data.rmObject.hParent = a.device;
  imported.object.data.rmObject.hObject = 0x102;
  rm_control(a.ctl, a.client, a.client,
             NV0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_FROM_FD, &imported,
             sizeof(imported));
  a.rm_memory = imported.object.data.rmObject.hObject;
  NV0000_CTRL_CLIENT_GET_HANDLE_INFO_PARAMS handle{};
  handle.hObject = a.rm_memory;
  handle.index = NV0000_CTRL_CMD_CLIENT_GET_HANDLE_INFO_INDEX_PARENT;
  rm_control(a.ctl, a.client, a.client, NV0000_CTRL_CMD_CLIENT_GET_HANDLE_INFO,
             &handle, sizeof(handle));
  require(handle.data.hResult == a.device, "RM memory parent mismatch");
  handle.index = NV0000_CTRL_CMD_CLIENT_GET_HANDLE_INFO_INDEX_CLASSID;
  rm_control(a.ctl, a.client, a.client, NV0000_CTRL_CMD_CLIENT_GET_HANDLE_INFO,
             &handle, sizeof(handle));
  require(handle.data.iResult == NV01_MEMORY_LOCAL_USER,
          "RM memory class mismatch");
  NV0000_CTRL_CLIENT_GET_ADDR_SPACE_TYPE_PARAMS address{};
  address.hObject = a.rm_memory;
  rm_control(a.ctl, a.client, a.client,
             NV0000_CTRL_CMD_CLIENT_GET_ADDR_SPACE_TYPE, &address,
             sizeof(address));
  require(address.addrSpaceType ==
              NV0000_CTRL_CMD_CLIENT_GET_ADDR_SPACE_TYPE_VIDMEM,
          "RM memory is not VIDMEM");
  NV0041_CTRL_SURFACE_INFO sizes[2]{};
  sizes[0].index = NV0041_CTRL_SURFACE_INFO_INDEX_PHYS_SIZE_LO;
  sizes[1].index = NV0041_CTRL_SURFACE_INFO_INDEX_PHYS_SIZE_HI;
  NV0041_CTRL_GET_SURFACE_INFO_PARAMS extent{};
  extent.surfaceInfoListSize = 2;
  extent.surfaceInfoList = reinterpret_cast<NvP64>(sizes);
  rm_control(a.ctl, a.client, a.rm_memory, NV0041_CTRL_CMD_GET_SURFACE_INFO,
             &extent, sizeof(extent));
  // RM may round the physical allocation beyond Vulkan's API allocation
  // size (for example, an 8 KiB buffer occupies a 64 KiB GPU page). Bound
  // that backing separately; never widen the export or CUDA logical view.
  backing_bytes = uint64_t(sizes[0].data) | (uint64_t(sizes[1].data) << 32);
  require(backing_bytes >= capacity && backing_bytes <= kMaxBytes &&
              backing_bytes % sysconf(_SC_PAGESIZE) == 0,
          "RM backing extent outside the bounded Vulkan allocation");
  nv_ioctl_export_to_dma_buf_fd_t exported{};
  exported.fd = -1;
  exported.hClient = a.client;
  exported.totalObjects = exported.numObjects = 1;
  exported.totalSize = bytes;
  exported.mappingType = NV_DMABUF_EXPORT_MAPPING_TYPE_DEFAULT;
  exported.handles[0] = a.rm_memory;
  exported.sizes[0] = bytes;
  exported.status = UINT32_MAX;
  bool ok = ioctl_ok(a.gpu_fd, NV_ESC_EXPORT_TO_DMABUF_FD, &exported,
                     sizeof(exported));
  a.dma_fd = exported.fd;  // Track even a partially returned descriptor.
  require(ok && exported.status == 0 && a.dma_fd >= 0,
          "RM core DMA-BUF export failed");
  require(fcntl(a.dma_fd, F_SETFD, FD_CLOEXEC) == 0, "DMA-BUF CLOEXEC failed");
  std::ifstream fdinfo("/proc/self/fdinfo/" + std::to_string(a.dma_fd));
  uint64_t actual = 0;
  std::string line, exporter;
  while (std::getline(fdinfo, line)) {
    std::istringstream fields(line);
    std::string key;
    fields >> key;
    if (key == "size:") fields >> actual;
    if (key == "exp_name:") fields >> exporter;
  }
  require(!fdinfo.bad() && actual == bytes && exporter == "nv_dmabuf",
          "DMA-BUF exporter/extent mismatch");
}

std::tuple<at::Tensor, int, std::string> allocate(uint64_t bytes, int ordinal) {
  require(!quarantined.load(), "admission sealed after unknown cleanup");
  long page = sysconf(_SC_PAGESIZE);
  require(page > 0 && bytes && bytes <= kMaxBytes && bytes % page == 0,
          "pool must contain whole host pages and be at most 1 GiB");
  c10::cuda::CUDAGuard guard(ordinal);
  // Torch may defer primary-context creation when selecting an unused device.
  // CUDA 12+ initializes it here, before the driver API borrows the context.
  auto status = cudaSetDevice(ordinal);
  require(status == cudaSuccess,
          "primary context initialization cudaError=" + std::to_string(status));
  auto owner = std::shared_ptr<Allocation>(new Allocation, release);
  CUdevice cuda_device;
  CUuuid uuid;
  cuda_check(cuCtxGetCurrent(&owner->context), "context");
  require(owner->context, "missing Torch CUDA context");
  cuda_check(cuCtxGetDevice(&cuda_device), "current device");
  cuda_check(cuDeviceGetUuid(&uuid, cuda_device), "UUID");
  char bdf[32]{};
  cuda_check(cuDeviceGetPCIBusId(bdf, sizeof(bdf), cuda_device), "BDF");
  VkDeviceSize capacity;
  uint64_t backing_bytes;
  create_vulkan(*owner, uuid, bytes, capacity);
  export_rm(*owner, uuid, bdf, capacity, bytes, backing_bytes);
  owner->cuda_fd = fcntl(owner->opaque, F_DUPFD_CLOEXEC, 3);
  require(owner->cuda_fd >= 0, "CUDA FD duplication failed");
  CUDA_EXTERNAL_MEMORY_HANDLE_DESC handle{};
  handle.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
  handle.handle.fd = owner->cuda_fd;
  handle.size = capacity;
  handle.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;
  cuda_check(cuImportExternalMemory(&owner->external, &handle),
             "OPAQUE_FD import");
  owner->cuda_fd = -1;  // Only successful import consumes this duplicate.
  CUDA_EXTERNAL_MEMORY_BUFFER_DESC buffer{};
  buffer.size = bytes;
  cuda_check(cuExternalMemoryGetMappedBuffer(&owner->pointer, owner->external,
                                             &buffer),
             "CUDA mapping");
  require(owner->pointer % page == 0, "CUDA mapping is not host-page aligned");
  unsigned enabled = 1, readback = 0;
  cuda_check(cuPointerSetAttribute(&enabled, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                   owner->pointer),
             "SYNC_MEMOPS set");
  cuda_check(cuPointerGetAttribute(&readback, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                   owner->pointer),
             "SYNC_MEMOPS query");
  require(readback == 1, "SYNC_MEMOPS not enabled");
  int ordering, flush;
  cuda_check(cuDeviceGetAttribute(
                 &ordering, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING,
                 cuda_device),
             "write ordering");
  cuda_check(
      cuDeviceGetAttribute(
          &flush, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS,
          cuda_device),
      "flush capabilities");
  std::ostringstream identity;
  identity << "driver=" << kVersion << " uuid=";
  for (unsigned char value : uuid.bytes) {
    constexpr char hex[] = "0123456789abcdef";
    identity << hex[value >> 4] << hex[value & 15];
  }
  identity << " bdf=" << bdf << " exporter=nv_dmabuf bytes=" << bytes
           << " allocation_bytes=" << capacity
           << " backing_bytes=" << backing_bytes
           << " sync_memops=1 ordering=" << ordering
           << " flush_options=" << flush;
  owner->identity = identity.str();
  auto tensor = at::from_blob(
      reinterpret_cast<void*>(owner->pointer), {static_cast<int64_t>(bytes)},
      [owner](void*) {},
      at::TensorOptions().dtype(at::kByte).device(at::kCUDA, ordinal));
  return {tensor, owner->dma_fd, owner->identity};
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("allocate", &allocate,
             "Owned CUDA byte tensor, borrowed DMA-BUF FD, identity");
}
