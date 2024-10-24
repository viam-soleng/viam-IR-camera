# Standard library
import cv2
from PIL import Image
from typing import Any, Dict, List, Mapping, Optional, Tuple, NamedTuple
from typing_extensions import Self
import numpy as np

# Viam module
from viam.components.camera import Camera, IntrinsicParameters, DistortionParameters
from viam.logging import getLogger
from viam.media.video import NamedImage, ViamImage
from viam.media.utils.pil import viam_to_pil_image, pil_to_viam_image, CameraMimeType
from viam.module.types import Reconfigurable, Stoppable
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName, ResponseMetadata
from viam.resource.base import ResourceBase
from viam.resource.types import Model, ModelFamily
import time

LOGGER = getLogger(__name__)

class IRCamera(Camera, Reconfigurable, Stoppable):
    family = ModelFamily("soleng-sandbox", "camera")
    MODEL = Model(family, "ir-camera")
    cap: Optional[cv2.VideoCapture] = None

    class Properties(NamedTuple):
        intrinsic_parameters: IntrinsicParameters
        """The properties of the camera"""
        distortion_parameters: DistortionParameters
        """The distortion parameters of the camera"""
        supports_pcd: bool = False
        """Whether the camera has a valid implementation of ``get_point_cloud``"""
    
    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        service = cls(config.name)
        service.cap = None
        service.validate(config)
        service.reconfigure(config, dependencies)
        return service

    @classmethod
    def validate(cls, config: ComponentConfig) -> None:
        return None

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> None:
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
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
        self.minimum_temperature = get_attribute_from_config("minimum_temperature", 16., float) # Celcius
        self.maximum_temperature = get_attribute_from_config("maximum_temperature", 25., float) # Celcius
        self.scale = get_attribute_from_config("scale", 1, int)
        self.blur_radius = get_attribute_from_config("blur_radius", 1, int)
        
        self.cap = cv2.VideoCapture(device_path, cv2.CAP_V4L)
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = round(int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))/2)

        LOGGER.info("SELF.WIDTH: {} | SELF.HEIGHT: {}".format(self.width, self.height))

    def stop(self, *,  extra: Optional[Mapping[str, Any]] = None, timeout: Optional[float] = None, **kwargs) -> None:
        LOGGER.info("STOPPING")
        self.cap.release()

    async def get_image(self, mime_type: str = "", *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None, **kwargs) -> ViamImage:
        if not self.cap.isOpened():
            LOGGER.error("video feed is not open, check if any ongoing processes are using the camera")
            return
        
        ret, frame = self.cap.read()

        if not ret:
            LOGGER.warn("No frame returned, retry")
            return

        imdata, thdata = np.array_split(frame, 2)

        thermal_image = self._create_thermal_image(thdata)
        img = Image.frombytes('RGB', (self.width * self.scale, self.height * self.scale), thermal_image)
        return pil_to_viam_image(img.convert('RGB'), CameraMimeType.JPEG)
    
    async def get_images(self, *, timeout: Optional[float] = None, **kwargs) -> Tuple[List[NamedImage], ResponseMetadata]:
        raise NotImplementedError()

    async def get_point_cloud(self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None, **kwargs) -> Tuple[bytes, str]:
        raise NotImplementedError()

    async def get_properties(self, *, timeout: Optional[float] = None, **kwargs) -> Properties:
        return self.camera_properties

    def _create_thermal_image(self, thdata):
        heatmap = np.zeros((self.height,self.width,1),dtype=np.uint8)
        start_time = time.time()

        min_temp = 1000
        max_temp = 0
        thdata = thdata.reshape(self.height, self.width, 2)
        hi = thdata[:, :, 0].astype(np.int32)
        lo = thdata[:, :, 1].astype(np.int32) * 256

        rawtemp = hi + lo
        temp = (rawtemp / 64) - 273.15

        min_temp = np.min(temp)
        max_temp = np.max(temp)

        scaled_temp = np.clip((temp - self.minimum_temperature) / (self.maximum_temperature - self.minimum_temperature) * 256, 0, 255).astype(np.uint8)
        heatmap[:, :, 0] = scaled_temp

        LOGGER.debug("Min temp: {} ({})| Max temp: {} ({})".format(min_temp, self.minimum_temperature, max_temp, self.maximum_temperature))

        #colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        resized_heatmap = cv2.resize(heatmap, (self.width * self.scale, self.height * self.scale,), interpolation=cv2.INTER_CUBIC)

        if self.blur_radius>0:
            resized_heatmap = cv2.blur(resized_heatmap,(self.blur_radius,self.blur_radius))
        colored_heatmap = cv2.cvtColor(resized_heatmap, cv2.COLOR_GRAY2RGB)
        
        end_time = time.time()
        execution_time = end_time - start_time
        LOGGER.debug(f"Thermal image creation took: {execution_time} seconds")
        
        return colored_heatmap
