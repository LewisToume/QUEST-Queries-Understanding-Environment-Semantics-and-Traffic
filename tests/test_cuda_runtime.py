import unittest
from unittest.mock import call, patch

import torch

from quest.cuda_runtime import initialize_cuda_device


class CudaRuntimeTest(unittest.TestCase):
    def test_init_and_selection_precede_allocator_statistics(self):
        with patch("quest.cuda_runtime.torch.cuda") as cuda:
            cuda.is_available.return_value = True
            cuda.device_count.return_value = 1
            device = initialize_cuda_device()
            self.assertEqual(device, torch.device("cuda:0"))
            self.assertEqual(cuda.mock_calls, [
                call.is_available(), call.init(), call.device_count(),
                call.set_device(0), call.reset_peak_memory_stats(device),
            ])

    def test_no_cuda_does_not_attempt_initialization_or_statistics(self):
        with patch("quest.cuda_runtime.torch.cuda") as cuda:
            cuda.is_available.return_value = False
            with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                initialize_cuda_device()
            cuda.init.assert_not_called()
            cuda.reset_peak_memory_stats.assert_not_called()

    def test_hidden_physical_device_is_not_used_as_a_logical_index(self):
        with patch("quest.cuda_runtime.torch.cuda") as cuda:
            cuda.is_available.return_value = True
            cuda.device_count.return_value = 1
            with self.assertRaisesRegex(RuntimeError, "visible_devices=1"):
                initialize_cuda_device(2)
            cuda.set_device.assert_not_called()
            cuda.reset_peak_memory_stats.assert_not_called()

    def test_driver_initialization_failure_is_not_hidden(self):
        with patch("quest.cuda_runtime.torch.cuda") as cuda:
            cuda.is_available.return_value = True
            cuda.init.side_effect = RuntimeError("driver initialization failed")
            with self.assertRaisesRegex(RuntimeError, "driver initialization failed"):
                initialize_cuda_device()
            cuda.reset_peak_memory_stats.assert_not_called()


if __name__ == "__main__":
    unittest.main()
