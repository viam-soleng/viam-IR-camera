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

LOGGER = getLogger(__name__)

class IRCamera(Camera, Reconfigurable, Stoppable):
    family = ModelFamily("viam-soleng", "camera")
    MODEL = Model(family, "ir-camera")

    class Properties(NamedTuple):
        intrinsic_parameters: IntrinsicParameters
        """The properties of the camera"""
        distortion_parameters: DistortionParameters
        """The distortion parameters of the camera"""
        supports_pcd: bool = True
        """Whether the camera has a valid implementation of ``get_point_cloud``"""
    
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
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = round(int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))/2)
        self.scale = 3
        self.contrast_alpha = 1.0
        self.blur_radius = 0
        print("SELF.WIDTH: {} | SELF.HEIGHT: {}".format(self.width, self.height))

    async def get_image(self, mime_type: str = "", *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None, **kwargs) -> ViamImage:
        if not self.cap.isOpened():
            print("CAP NOT OPENED")
            return
        
        ret, frame = self.cap.read()

        if not ret:
            print("BAD RETURN")
            return

        # print("pre split")
        # print(frame.shape)
        image, bot = np.array_split(frame, 2)

        # print("post split")
        # print(image.shape)
        thermal_image = await self.create_thermal_image(image)
        # print("thermal image")
        # print(thermal_image.shape)

        #print("required size = {},{}".format(self.width, self.height))
        img = Image.frombytes('RGB', (self.width * self.scale, self.height * self.scale), thermal_image)
        return pil_to_viam_image(img.convert('RGB'), CameraMimeType.JPEG)
    
    async def get_images(self, *, timeout: Optional[float] = None, **kwargs) -> Tuple[List[NamedImage], ResponseMetadata]:
        raise NotImplementedError()

    async def get_point_cloud(self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None, **kwargs) -> Tuple[bytes, str]:
        raise NotImplementedError()

    async def get_properties(self, *, timeout: Optional[float] = None, **kwargs) -> Properties:
        return self.camera_properties

    async def create_thermal_image(self, imdata):
        converted_image = cv2.cvtColor(imdata,  cv2.COLOR_YUV2BGR_YUYV)
        scaled_image = cv2.convertScaleAbs(converted_image, alpha=self.contrast_alpha)
        resize_image = cv2.resize(scaled_image, (self.width * self.scale, self.height * self.scale), interpolation=cv2.INTER_CUBIC)#Scale up!
       
        if self.blur_radius>0:
            resize_image = cv2.blur(resize_image,(self.blur_radius,self.blur_radius))   

        heatmap = cv2.applyColorMap(resize_image, cv2.COLORMAP_JET)

        return heatmap

# def get_temperature_data(thdata):
        
#         # Find max temperature
#         lomax = thdata[...,1].max()
#         posmax = thdata[...,1].argmax()

# 		mcol,mrow = divmod(posmax, self.width)
# 		himax = thdata[mcol][mrow][0]
# 		lomax=lomax*256
        
# 		maxtemp = himax+lomax
# 		maxtemp = convert_kelvin_to_celcius(maxtemp)
#         print("Max temperature: " + str(maxtemp))

#         # Find min temperature
#         lomin = thdata[...,1].min()
# 		posmin = thdata[...,1].argmin()

# 		lcol,lrow = divmod(posmin, self.width)
# 		himin = thdata[lcol][lrow][0]
# 		lomin=lomin*256

# 		mintemp = himin+lomin
# 		mintemp = convert_kelvin_to_celcius(mintemp)
#         print("Min temperature: " + str(mintemp))

#         # Find average temperature
# 		loavg = thdata[...,1].mean()
# 		hiavg = thdata[...,0].mean()
# 		loavg=loavg*256

# 		avgtemp = loavg+hiavg
# 		avgtemp = convert_kelvin_to_celcius(avgtemp)

#         print("Avg temperature: " + str(avgtemp))

#         return (maxtemp, mintemp, avgtemp)

# def convert_kelvin_to_celcius(temp):
#     	return round((temp/64)-273.15, 2)