# Standard library
import cv2
from PIL import Image
from digitalio import DigitalInOut
from adafruit_vl53l0x import VL53L0X
from typing import Any, Dict, List, Mapping, Optional, Tuple
from typing_extensions import Self

# Viam module
from viam.components.board import Board
from viam.components.camera import Camera, Properties
from viam.logging import getLogger
from viam.media.video import NamedImage, ViamImage
from viam.media.utils.pil import viam_to_pil_image, pil_to_viam_image, CameraMimeType
from viam.module.types import Reconfigurable, Stoppable
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName, ResponseMetadata
from viam.resource.base import ResourceBase
from viam.resource.types import Model, ModelFamily

LOGGER = getLogger(__name__)

class TOFSensor(Camera, Reconfigurable, Stoppable):
    family = ModelFamily("viam-soleng", "camera")
    MODEL = Model(family, "ir-camera")
    
    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        service = cls(config.name)
        service.validate(config)
        service.reconfigure(config, dependencies)
        return service

    @classmethod
    def validate(cls, config: ComponentConfig) -> None:
        return None

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> None:

        def get_attribute_from_config(attribute_name: str, default, of_type=None):
            if attribute_name not in config.attributes.fields:
                return default

            if default is None:
                if of_type is None:
                    raise Exception(
                        "If default value is None, of_type argument can't be empty"
                    )
                type_default = of_type
            else:
                type_default = type(default)

            if type_default == bool:
                return config.attributes.fields[attribute_name].bool_value
            elif type_default == int:
                return int(config.attributes.fields[attribute_name].number_value)
            elif type_default == float:
                return config.attributes.fields[attribute_name].number_value
            elif type_default == str:
                return config.attributes.fields[attribute_name].string_value
            elif type_default == list:
                return list(config.attributes.fields[attribute_name].list_value)
            elif type_default == dict:
                return dict(config.attributes.fields[attribute_name].struct_value)
    
        # # Extract config info
        device_path = get_attribute_from_config("device_path", None, str)

        self.cap = cv2.VideoCapture(device_path, cv2.CAP_V4L)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # cv2.namedWindow('Thermal',cv2.WINDOW_GUI_NORMAL)
        # font=cv2.FONT_HERSHEY_SIMPLEX

    async def get_image(self, mime_type: str = "", *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None, **kwargs) -> ViamImage:
        ret, frame = self.cap.read()

        if ret:
            img = Image.frombytes('RGB', (self.width, self.height), frame)
            return pil_to_viam_image(img.convert('RGB'), CameraMimeType.JPEG)
    
    async def get_images(self, *, timeout: Optional[float] = None, **kwargs) -> Tuple[List[NamedImage], ResponseMetadata]:
        raise NotImplementedError()

    async def get_point_cloud(self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None, **kwargs) -> Tuple[bytes, str]:
        raise NotImplementedError()

    async def get_properties(self, *, timeout: Optional[float] = None, **kwargs) -> Properties:
        return self.camera_properties
    