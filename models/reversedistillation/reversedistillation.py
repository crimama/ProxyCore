from ..anomalib.models.reverse_distillation.torch_model import ReverseDistillationModel
from ..anomalib.models.reverse_distillation.loss import ReverseDistillationLoss
from ..anomalib.models.reverse_distillation.anomaly_map import AnomalyMapGenerationMode
from torch import Tensor, nn 

class ReverseDistillation(ReverseDistillationModel):
    def __init__(self,
        backbone: str,
        input_size ,
        layers ,
        anomaly_map_mode: AnomalyMapGenerationMode,
        pre_trained: bool = True) -> None:
        
        super(ReverseDistillation,self).__init__(
            backbone         = backbone,
            input_size       = input_size,
            layers           = layers,
            anomaly_map_mode = anomaly_map_mode,
            pre_trained      = pre_trained
        )
        
        self._criterion = ReverseDistillationLoss()        
        
    def forward(self, images: Tensor):
        self.encoder.eval()

        if self.tiler:
            images = self.tiler.tile(images)
        encoder_features = self.encoder(images)
        encoder_features = list(encoder_features.values())
        decoder_features = self.decoder(self.bottleneck(encoder_features))

        if self.tiler:
            for i, features in enumerate(encoder_features):
                encoder_features[i] = self.tiler.untile(features)
            for i, features in enumerate(decoder_features):
                decoder_features[i] = self.tiler.untile(features)
                
        output = (encoder_features, decoder_features)
        
        return output 
    
    
    # def forward(self, input_tensor: Tensor):
    #     outputs = self._forward(input_tensor)
    #     return outputs
    
    def criterion(self, outputs: tuple):
        (encoder_features, decoder_features) = outputs
        
        loss = self._criterion(encoder_features, decoder_features)
        
        return loss 
    
    def get_score_map(self, outputs: tuple):
        (encoder_features, decoder_features) = outputs 
        output = self.anomaly_map_generator(encoder_features, decoder_features)
        return output
        
        