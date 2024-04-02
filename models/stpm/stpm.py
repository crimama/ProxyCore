import timm 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class STPM(nn.Module):
    def __init__(self, model_name:str,  input_size:list, layer:list):
        super(STPM,self).__init__()

        self.teacher = self._create_model(model_name = model_name, pretrained= True)
        self.student = self._create_model(model_name = model_name, pretrained= False)
        self.layer = [str(l+4) for l in layer]                    
        self.mse_loss = nn.MSELoss(reduction="sum")
        self.input_size = input_size
        
    def _create_model(self, model_name: str, pretrained: bool):
        model = timm.create_model(model_name = model_name, pretrained = pretrained)
        model = nn.Sequential(*list(model.children())[:-2])
        
        if pretrained:
            # teacher required grad False  
            for p in model.parameters():
                p.requires_grad = False 
            model.training = False 
        return model 
    
    def train(self):
        self.student.training = True 
        
    def eval(self):
        self.student.training = False 
            
    def _forward(self, x) -> list:
        t_features = [] 
        s_features = [] 
        
        x_s = x.clone()
        x_t = x.clone() 
        for (t_name, t_module), (s_name,s_module) in zip(self.teacher._modules.items(), self.student._modules.items()):
            x_t = t_module(x_t)
            x_s = s_module(x_s)
            
            if t_name in self.layer:
                t_features.append(x_t)
                s_features.append(x_s)
                
        return t_features, s_features
    
    def _criterion(self, t_features: list, s_features: list):
        total_loss = 0 
        for t,s in zip(t_features, s_features):
            #! Full_base 버전 
            t,s = F.normalize(t, dim=1), F.normalize(s, dim=1)
            layer_loss = torch.sum((t.type(torch.float32) - s.type(torch.float32)) ** 2, 1).mean()
            
            #! Anomalib 버전 
            # height, width = t.shape[2:]
            # norm_teacher_features = F.normalize(t)
            # norm_student_features = F.normalize(s)
            # layer_loss = (0.5 / (width * height)) * self.mse_loss(norm_teacher_features, norm_student_features)
            
            total_loss += layer_loss
        return total_loss 

    def forward(self, x, only_loss:bool = True):
        t_features, s_features = self._forward(x)
        loss = self._criterion(t_features, s_features)
        
        if only_loss:
            return loss 
        else:
            return loss, [t_features, s_features]
        
    def get_score_map(self, outputs: list) -> torch.Tensor:
        '''
        outputs = [t_outputs, s_outputs]
        sm.shape = (B,1,64,64)
        '''
        t_outputs, s_outputs = outputs[0], outputs[1]
        score_map = 1.
        for t, s in zip(t_outputs, s_outputs):
            t,s = F.normalize(t,dim=1),F.normalize(s,dim=1) # channel wise normalize 
            sm = torch.sum((t - s) ** 2, 1, keepdim=True) # channel wise average 
            sm = F.interpolate(sm, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False) # Intepolation : (1,w,h) -> (1,64,64)
            score_map = score_map * sm 
        return score_map
            
    