import argparse
import torch
import os
from PIL import Image
import numpy as np
from typing import List, Optional, Union
from enum import Enum
import re
import sys
import random
sys.path.append('/mnt/hpfs/baaiei/habitat/LLaVA-NeXT-main')
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import KeywordsStoppingCriteria
from llava.mm_utils import get_model_name_from_path

class ActionType(Enum):
    FORWARD = "FORWARD"
    TURN_LEFT = "TURN_LEFT" 
    TURN_RIGHT = "TURN_RIGHT"
    STOP = "STOP"
    UNKNOWN = "UNKNOWN"

    def to_action_id(self) -> int:
        """Convert ActionType to numeric action ID.
        
        Returns:
            int: Action ID where:
                FORWARD -> 1
                TURN_LEFT -> 2
                TURN_RIGHT -> 3
                Others (STOP/UNKNOWN) -> random choice from [1,2,3]
        """
        action_map = {
            ActionType.FORWARD: 1,
            ActionType.TURN_LEFT: 2,
            ActionType.TURN_RIGHT: 3,
            ActionType.STOP: 0
        }
        return action_map.get(self, random.choice([1, 2, 3]))

class LLaVANavigationInference:
    def __init__(
        self,
        model_dir: str = "/home/vlm/workspace/checkpoints/resume_finetune_Llava-Onevision-qwen2.5-cvpr-exp0/checkpoint-6000",
        conv_mode: str = "qwen_2",
        temperature: float = 0.2,
        top_p: float = 0.95,
        num_beams: int = 1,
        max_new_tokens: int = 512,
        navigation_prompt: str = "What action should I take based on the current view?",
    ):
        """Initialize the LLaVA navigation inference model.

        Args:
            model_dir: Path to the model checkpoint directory
            conv_mode: Conversation mode ("qwen_2" by default)
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            num_beams: Number of beams for beam search
            max_new_tokens: Maximum number of new tokens to generate
            navigation_prompt: Default prompt for navigation tasks
        """
        self.config = {
            "model_dir": model_dir,
            "conv_mode": conv_mode,
            "temperature": temperature,
            "top_p": top_p,
            "num_beams": num_beams,
            "max_new_tokens": max_new_tokens,
        }
        self.navigation_prompt = navigation_prompt
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the LLaVA model, tokenizer, and image processor."""
        disable_torch_init()
        model_name = get_model_name_from_path(self.config["model_dir"])
        assert "llama" in model_name.lower() or "qwen" in model_name.lower(), \
            "model_name should contain 'llama' or 'qwen'"
            
        model_name = model_name.lower().replace("llama", "llava_llama").replace("qwen", "llava_qwen")
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            self.config["model_dir"], 
            None, 
            model_name, 
            torch_dtype='bfloat16'
        )
        print(self.model)
        # exit()
    def _preprocess_qwen(
        self,
        prompt: str,
        conversation_history: list = [],
        has_image: bool = True,
        system_message: str = "You are a helpful navigation assistant."
    ):
        """Preprocess the conversation for Qwen model format."""
        sources = []
        # print(conversation_history)
        # exit()
        if len(conversation_history)>0:
            # print(conversation_history)
            for hist in conversation_history:
                sources.append({"from": hist["from"], "value": hist["value"]})
        sources.append({"from": "human", "value": prompt})

        # sources = [
        #     {"from": "human", "value": prompt}
        # ]
        
        roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
        input_ids, targets = [], []
        input_id, target = [], []
        
        # Add system message
        system = [self.tokenizer.additional_special_tokens_ids[0]] + \
                self.tokenizer("system").input_ids + \
                self.tokenizer(system_message).input_ids + \
                [self.tokenizer.additional_special_tokens_ids[1]] + \
                self.tokenizer("\n").input_ids
        input_id += system
        target += [IGNORE_INDEX] * len(system)
        
        # Process prompt
        for sentence in sources:
            role = roles[sentence["from"]]
            if has_image and "<image>" in sentence["value"]:
                texts = sentence["value"].split("<image>")
                _input_id = self.tokenizer(role).input_ids + self.tokenizer("\n").input_ids
                for i, text in enumerate(texts):
                    _input_id += self.tokenizer(text).input_ids
                    if i < len(texts) - 1:
                        _input_id += [IMAGE_TOKEN_INDEX] + self.tokenizer("\n").input_ids
                _input_id += [self.tokenizer.additional_special_tokens_ids[1]] + self.tokenizer("\n").input_ids
            else:
                _input_id = self.tokenizer(role).input_ids + \
                           self.tokenizer("\n").input_ids + \
                           self.tokenizer(sentence["value"]).input_ids + \
                           [self.tokenizer.additional_special_tokens_ids[1]] + \
                           self.tokenizer("\n").input_ids
            input_id += _input_id
            target += [IGNORE_INDEX] * len(_input_id)

        input_ids.append(input_id)
        targets.append(target)
        return torch.tensor(input_ids, dtype=torch.long)

    def _parse_action(self, text: str) -> ActionType:
        """Parse the navigation action from the model's response."""
        text = text.lower()
        patterns = {
            ActionType.FORWARD: [
                r"forward",
                r"move\s+forward",
                r"go\s+forward",
                r"walk\s+forward",
                r"proceed",
                r"advance",
                r"straight",
                r"go\s+straight",
                r"move\s+ahead",
                r"walk\s+ahead",
                r"continue",
                r"move\s+straight",
                r"go\s+ahead",
            ],
            ActionType.TURN_LEFT: [
                r"turn\s*left",
                r"turn\s+to\s+(\w+\s+)?left",
                r"rotate\s*left",
                r"left\s+turn",
                r"go\s+left",
                r"move\s+left",
                r"turn_left",
                r"turn-left",
            ],
            ActionType.TURN_RIGHT: [
                r"turn\s*right",
                r"turn\s+to\s+(\w+\s+)?right",
                r"rotate\s*right",
                r"right\s+turn",
                r"go\s+right",
                r"move\s+right",
                r"turn_right",
                r"turn-right",
            ],
            ActionType.STOP: [
                r"stop",
                r"halt",
                r"wait",
                r"stand\s+still",
                r"stay",
                r"pause",
                r"end",
                r"finish",
                r"terminate",
            ]
        }
        
        for action_type, pattern_list in patterns.items():
            combined_pattern = '|'.join(f"({pattern})" for pattern in pattern_list)
            if re.search(combined_pattern, text, re.IGNORECASE):
                return action_type
                
        return ActionType.UNKNOWN

    def _process_image(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """Process different types of image inputs into tensor format.
        
        Args:
            image: Can be:
                - Path to image file (str)
                - Numpy array (np.ndarray)
                - PIL Image (Image.Image)
                
        Returns:
            torch.Tensor: Processed image tensor
        """
        if isinstance(image, str):
            # PIL.Image.open() 默认格式
            image = Image.open(image)
            # print(111)
        
        elif isinstance(image, np.ndarray):
            # Numpy array 预处理
            if image.dtype != np.uint8:
                # 确保值范围在 0-255
                if image.dtype == np.float32 or image.dtype == np.float64:
                    image = np.clip(image * 255, 0, 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # 确保是三通道 RGB
            if len(image.shape) == 2:  # 灰度图像
                image = np.stack([image] * 3, axis=-1)
            elif len(image.shape) == 3:
                if image.shape[-1] == 1:  # 单通道图像带维度
                    image = np.concatenate([image] * 3, axis=-1)
                elif image.shape[-1] != 3:
                    raise ValueError(f"Expected 3 channels, got {image.shape[-1]}")
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")
            
            # 转换为 PIL Image
            image = Image.fromarray(image, mode='RGB')
            
        elif isinstance(image, Image.Image):
            if image.mode != 'RGB':
                image = image.convert('RGB')
        else:
            raise ValueError("Image must be either a file path, numpy array, or PIL Image")


        # Process through image processor
        image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
        return image_tensor.to(dtype=torch.bfloat16, device="cuda")

    def get_action(
        self,
        images: Union[str, np.ndarray, Image.Image, List[Union[str, np.ndarray, Image.Image]]],
        conversation_history = [],
        prompt: Optional[str] = None,
    ) -> ActionType:
        """Get navigation action from input images.
        
        Args:
            images: Can be single image or list of images. Each image can be:
                   - Path to image file (str)
                   - Numpy array (np.ndarray)
                   - PIL Image (Image.Image)
            prompt: Custom prompt (uses default navigation prompt if None)
            
        Returns:
            ActionType: The parsed navigation action
        """
        # Handle single image input
        if not isinstance(images, list):
            images = [images]
            
        # Process all images
        image_tensors = [self._process_image(img) for img in images]
        
        # self.conversation_history.append({"from": "human", "value": user_input})
        # Get model input
        # if hasattr(self.tokenizer, 'additional_special_tokens'):
        #     print(self.tokenizer.additional_special_tokens)
        #     image_tokens = [token for token in self.tokenizer.additional_special_tokens if '<image>' in token]
        #     print(f"\nImage-related special tokens: {image_tokens}")
        input_ids = self._preprocess_qwen(prompt, conversation_history,has_image=True).cuda()

        # print(f"\nInput tokens shape: {input_ids.shape}")
        # print(f"Number of image tokens in input: {(input_ids == IMAGE_TOKEN_INDEX).sum().item()}")
        # exit()
        # Setup stopping criteria
        conv_mode = conv_templates[self.config["conv_mode"]]
        stop_str = conv_mode.sep if conv_mode.sep_style != SeparatorStyle.TWO else conv_mode.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)
        # print(image_tensors[0].shape)
        # Generate response
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensors,
                do_sample=True if self.config["temperature"] > 0 else False,
                temperature=self.config["temperature"],
                top_p=self.config["top_p"],
                num_beams=self.config["num_beams"],
                max_new_tokens=self.config["max_new_tokens"],
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )
            
        # Process response
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        response = outputs.replace("assistant\n", "").split(stop_str)[0].strip()
        # Parse and return action
        return self._parse_action(response), response
    
    def get_action_id(
        self,
        images: Union[str, np.ndarray, Image.Image, List[Union[str, np.ndarray, Image.Image]]],
        conversation_history=[],
        prompt: Optional[str] = None,
    ) -> int:
        """Get only the numeric action ID from input images.
        
        Args:
            images: Can be single image or list of images. Each image can be:
                   - Path to image file (str)
                   - Numpy array (np.ndarray)
                   - PIL Image (Image.Image)
            prompt: Custom prompt (uses default navigation prompt if None)
            
        Returns:
            int: Action ID (1 for forward, 2 for left, 3 for right, 
                 random choice for unknown/stop)
        """
        
        action,response = self.get_action(images, conversation_history,prompt)
            
        return action.to_action_id(),response
# def main():
#     # Example usage
#     model = LLaVANavigationInference()
    
#     # Example with file path
#     action = model.get_action("/path/to/image.jpg")
#     print(f"Action from file: {action.value}")
    
#     # Example with numpy array
#     img_array = np.random.rand(224, 224, 3).astype(np.float32)  # Example array
#     action = model.get_action(img_array)
#     print(f"Action from array: {action.value}")
    
#     # Example with multiple images
#     images = [
#         "/path/to/image1.jpg",
#         np.random.rand(224, 224, 3).astype(np.float32)
#     ]
#     action = model.get_action(images)
#     print(f"Action from multiple images: {action.value}")

# if __name__ == "__main__":
#     main()