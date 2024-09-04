# Thermal IR Camera (TOPDON)

This is a [Viam module](https://docs.viam.com/manage/configuration/#modules) for [TOPDON](https://www.topdon.com/)'s family of thermal imagers. This module extracts the raw intensity and thermal map from the video feed using the v4l2-ctl video capture package and converts it into a gray scale image using a defined minimum and maximum temperature. 

This viam-kuka module is particularly useful in applications that require a Kuka arm to be operated in conjunction with other resources (such as cameras, sensors, actuators, CV) offered by the [Viam Platform](https://www.viam.com/) and/or separate through your own code. 

> [!NOTE]
> For more information on modules, see [Modular Resources](https://docs.viam.com/registry/#modular-resources).

## Configure your TOPDON Thermal Camera

> [!NOTE]
> Before configuring your Kuka Arm, you must [add a machine](https://docs.viam.com/fleet/machines/#add-a-new-machine).

Navigate to the **CONFIGURE** tab of your machine’s page in [the Viam app](https://app.viam.com/). Click the **+** icon next to your machine part in the left-hand menu and select **Component**. Select the `camera` type, then search for and select the `camera / viam-ir-camera` model. Click **Add module**, then enter a name or use the suggested name for your arm and click **Create**.

On the new component panel, copy and paste the following attribute template into your arm’s attributes field:

```json
{
  "device_path": "/dev/video0",
  "minimum_temperature": 10,
  "maximum_temperature": 30,
  "scale": 2,
  "blur_radius": 1
}
```

> [!NOTE]
> For more information, see [Configure a Machine](https://docs.viam.com/build/configure/).

## Attributes

The following attributes are available:

| Name | Type | Inclusion | Default | Description |
| ---- | ---- | --------- | ------- | ----------- |
| `device_path` | string | **Required** | N/A | The video path to the device.  |
| `minimum_temperature` | float64 | Optional | 16.0 | The minimum temperature expected in the image. This will be used to scale the resultant image accordingly. Any values below this bound will be set to the minimum. |
| `maximum_temperature` | float64 | Optional | 25.0 | The maximum temperature expected in the image. This will be used to scale the resultant image accordingly. Any values above this bound will be set to the maximum. |
| `scale` | int | Optional | 1 | Used to scale the image.  |
| `blur_radius` | int | Optional | 1 | Applies a blur to the resultant image using the provided radius. |

## Known Supported Hardware

Support for the following devices has been confirmed.

| Devices             | Mac OSX |  Linux (aarch64)  |  Linux (amd64)  |
|---------------------|---------|-------------------|-----------------|
| TOPDON TC001        |    X    |          X        |         X       |

