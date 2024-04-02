import torch 
from torch import Tensor, nn
from ..anomalib.models.fastflow.torch_model import FastflowModel

from timm.models.vision_transformer import VisionTransformer
from timm.models.cait import Cait



class FastFlow(FastflowModel):
    def __init__(
        self,
        input_size,
        backbone: str,
        pre_trained: bool = True,
        flow_steps: int = 8,
        conv3x3_only: bool = False,
        hidden_ratio: float = 1.0,
        ):
        super(FastFlow,self).__init__(
            input_size   = input_size,
            backbone     = backbone,
            pre_trained  = pre_trained,
            flow_steps   = flow_steps,
            conv3x3_only = conv3x3_only,
            hidden_ratio = hidden_ratio
        )
        
    def _forward(self, input_tensor: Tensor):
        """Forward-Pass the input to the FastFlow Model.

        Args:
            input_tensor (Tensor): Input tensor.

        Returns:
            Tensor | list[Tensor] | tuple[list[Tensor]]: During training, return
                (hidden_variables, log-of-the-jacobian-determinants).
                During the validation/test, return the anomaly map.
        """

        self.feature_extractor.eval()
        if isinstance(self.feature_extractor, VisionTransformer):
            features = self._get_vit_features(input_tensor)
        elif isinstance(self.feature_extractor, Cait):
            features = self._get_cait_features(input_tensor)
        else:
            features = self._get_cnn_features(input_tensor)

        # Compute the hidden variable f: X -> Z and log-likelihood of the jacobian
        # (See Section 3.3 in the paper.)
        # NOTE: output variable has z, and jacobian tuple for each fast-flow blocks.
        hidden_variables: list[Tensor] = []
        log_jacobians: list[Tensor] = []
        for fast_flow_block, feature in zip(self.fast_flow_blocks, features):
            hidden_variable, log_jacobian = fast_flow_block(feature)
            hidden_variables.append(hidden_variable)
            log_jacobians.append(log_jacobian)

        return_val = (hidden_variables, log_jacobians)
        return return_val
        
    def forward(self, input_tensor: Tensor):
        return_val = self._forward(input_tensor)

        return return_val
    
    def criterion(self, outputs):
        (hidden_variables, jacobians) = outputs
        
        loss = torch.tensor(0.0, device=hidden_variables[0].device)
        for hidden_variable, jacobian in zip(hidden_variables, jacobians):
            loss += torch.mean(0.5 * torch.sum(hidden_variable**2, dim=(1, 2, 3)) - jacobian)
        return loss 
    
    def get_score_map(self, x):
        (hidden_variables, log_jacobians) = x
        return_val = self.anomaly_map_generator(hidden_variables)
        
        return return_val