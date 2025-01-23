from our_model.MsC2A_VMUnet import *
from modelszoo.Umamba import *
from modelszoo.AC_Mamba import *
from modelszoo.VM_UNet import *
from modelszoo.Swin_UNet import *
from modelszoo.MISSFormer import *
from modelszoo.H2Former import *
from modelszoo.VM_UNet2 import *
from modelszoo.R2UNet import *
from modelszoo.H_vmunet import *
from modelszoo.unetpp import *
from modelszoo.Trans_UNet import *
from modelszoo.attention_unet import *


class ObjectCreator:
    def __init__(self, args, cls) -> None:
        self.args = args
        self.cls_net = cls
    def __call__(self):
        return self.cls_net(**self.args)


models = {
    "MsC2A_VMUnet": ObjectCreator(cls=MsC2A_VMUnet, args=dict(

    )),
    # "MISSFormer":ObjectCreator(cls=MISSFormer,args=dict(
    #     num_classes=1
    # )),
    # "UNetpp": ObjectCreator(cls=ResNet34UnetPlus, args=dict(
    #
    # )),
    # "R2U_Net":ObjectCreator(cls=R2U_Net,args=dict(
    #     img_ch=1, output_ch=1
    # )),
    # "Swin_Unet":ObjectCreator(cls=SwinUnet,args=dict(
    #     num_classes=1,img_size=224
    # )),
    # "U-Mmaba":ObjectCreator(cls=UltraLight_VM_UNet,args=dict(
    #     num_classes=1, input_channels=1
    # )),
    # "AC-Mamba":ObjectCreator(cls=AC_MambaSeg,args=dict(
    #
    # )),
    # "VM-UNet":ObjectCreator(cls=VMUNet,args=dict(
    #     input_channels=1
    # )),
    # "H2Former": ObjectCreator(cls=res34_swin_MS, args=dict(
    #     image_size=224, num_class=1
    # )),
    # "VM_Unetv2":ObjectCreator(cls=VMUNetV2,args=dict(
    #     input_channels=1,num_classes=1
    # )),
    # "H_vmunet":ObjectCreator(cls=H_vmunet,args=dict(
    #     num_classes=1, input_channels=1
    # ))
    # "Trans_UNet":ObjectCreator(cls=TransUNet,args=dict(
    #     num_classes=1, input_channels=1
    # ))
    # "attention_unet":ObjectCreator(cls=AttU_Net,args=dict(
    #     num_classes=1, input_channels=1
    # ))

    # More models can be added here......
}

