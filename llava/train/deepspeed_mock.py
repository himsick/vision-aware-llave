# Save as deepspeed_mock.py
class Mock:
    def __getattr__(self, name): return Mock()
    def __call__(self, *args, **kwargs): return Mock()

import sys
sys.modules["deepspeed"] = Mock()
sys.modules["deepspeed.zero"] = Mock()
sys.modules["deepspeed.runtime"] = Mock()
sys.modules["deepspeed.runtime.zero"] = Mock()
print("DeepSpeed has been mocked for single-GPU training.")