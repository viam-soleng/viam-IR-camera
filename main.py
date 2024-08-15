import asyncio

from viam.components.camera import Camera
from viam.logging import getLogger
from viam.module.module import Module
from viam.resource.registry import Registry, ResourceCreatorRegistration
from src.ir_camera import IRCamera

LOGGER = getLogger(__name__)

async def main():
    """This function creates and starts a new module, after adding all desired resources.
    Resources must be pre-registered. For an example, see the `__init__.py` file.
    """
    Registry.register_resource_creator(
        Camera.SUBTYPE,
        IRCamera.MODEL,
        ResourceCreatorRegistration(IRCamera.new, IRCamera.validate),
    )

    module = Module.from_args()
    module.add_model_from_registry(IRCamera.SUBTYPE, IRCamera.MODEL)
    LOGGER.debug("Starting module in main.py.")
    await module.start()


if __name__ == "__main__":
    asyncio.run(main())
