import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import copy

class StyleTransfer:
    def __init__(self):
        # Use MPS (Metal Performance Shaders) for M1 Macs
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Default layers for content and style representation
        self.content_layers_default = ['conv_4']
        self.style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        
        try:
            # Load VGG19 with batch normalization for better style transfer
            self.cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(self.device).eval()
            
            # Freeze all parameters
            for param in self.cnn.parameters():
                param.requires_grad_(False)
            
            # Normalization parameters
            self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
            self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
            
            # Define layers for style and content representation
            # Using deeper layers for better style capture
            self.content_layers = self.content_layers_default
            self.style_layers = self.style_layers_default
            
            # Layer weights for style (deeper layers have more weight)
            self.style_layer_weights = [0.1, 0.2, 0.3, 0.2, 0.2]
            
            print("Enhanced model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def load_image(self, image_path, imsize=512):
        """Load an image and convert it to a PyTorch tensor."""
        image = Image.open(image_path).convert('RGB')
        
        # Resize the image if needed
        if imsize is not None:
            transform = transforms.Compose([
                transforms.Resize(imsize),
                transforms.CenterCrop(imsize),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
        # Add batch dimension
        image = transform(image).unsqueeze(0)
        return image.to(self.device, torch.float)

    def save_image(self, tensor, filename):
        """Convert a PyTorch tensor to an image and save it."""
        # Clone the tensor to not change the original
        image = tensor.cpu().clone()
        # Remove the batch dimension
        image = image.squeeze(0)
        # Convert to PIL image
        unloader = transforms.ToPILImage()
        image = unloader(image)
        image.save(filename)

    def get_style_model_and_losses(self, style_img, content_img, 
                                 content_layers=None, style_layers=None):
        """Build the style transfer model and compute the losses."""
        content_layers = content_layers or self.content_layers_default
        style_layers = style_layers or self.style_layers_default
        
        # Create a copy of the VGG model
        cnn = copy.deepcopy(self.cnn)
        
        # Build the model
        model = nn.Sequential()
        content_losses = []
        style_losses = []
        
        # Assuming cnn is a nn.Sequential
        i = 0  # Incremented every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f'conv_{i}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{i}'
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{i}'
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn_{i}'
            else:
                raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')
                
            model.add_module(name, layer)
            
            if name in content_layers:
                # Add content loss
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module(f'content_loss_{i}', content_loss)
                content_losses.append(content_loss)
                
            if name in style_layers:
                # Add style loss
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module(f'style_loss_{i}', style_loss)
                style_losses.append(style_loss)
                
        # Now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
                
        model = model[:(i + 1)]
        
        return model, style_losses, content_losses

    def run_style_transfer(self, content_img, style_img, input_img, num_steps=300,
                          style_weight=1000000, content_weight=1, tv_weight=1e-5):
        """Run the style transfer with improvements for better quality."""
        model, style_losses, content_losses = self.get_style_model_and_losses(
            style_img, content_img)
        
        # We want to optimize the input image
        input_img.requires_grad_(True)
        model.requires_grad_(False)
        
        # Use Adam optimizer with weight decay for better convergence
        optimizer = optim.Adam([input_img], lr=0.02)  # Slightly higher learning rate for faster convergence
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)  # Learning rate decay
        
        # Track best result
        best_loss = float('inf')
        best_result = input_img.detach().clone()
        
        for step in range(num_steps):
            def closure():
                # Clamp the image to maintain valid pixel values
                with torch.no_grad():
                    input_img.data.clamp_(0, 1)
                
                optimizer.zero_grad()
                model(input_img)
                
                # Calculate style and content losses
                style_score = 0
                content_score = 0
                
                # Weighted sum of style losses
                for i, sl in enumerate(style_losses):
                    if i < len(self.style_layer_weights):
                        style_score += sl.loss * self.style_layer_weights[i]
                    else:
                        style_score += sl.loss * 0.1  # Default weight if not specified
                
                # Content loss
                for cl in content_losses:
                    content_score += cl.loss
                
                # Add total variation regularization for smoother results
                tv_loss = self.total_variation_regularization(input_img)
                
                # Weighted loss
                style_score *= style_weight
                content_score *= content_weight
                tv_loss *= tv_weight
                
                total_loss = style_score + content_score + tv_loss
                total_loss.backward()
                
                return total_loss
            
            # Update weights
            loss = optimizer.step(closure)
            scheduler.step()
            
            # Track best result
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_result = input_img.detach().clone()
            
            # Print progress
            if step % 25 == 0 or step == num_steps - 1:
                print(f"Step {step + 1}/{num_steps}:")
                print(f"  Total Loss: {loss.item():.2f}, "
                      f"Style: {style_weight * sum(sl.loss.item() for sl in style_losses):.2f}, "
                      f"Content: {content_weight * sum(cl.loss.item() for cl in content_losses):.2f}")
        
        # Return the best result found during optimization
        return best_result
    
    def total_variation_regularization(self, img, weight=1.0):
        """Total variation regularization to reduce noise in the output."""
        # Calculate the difference between neighboring pixel-values.
        # The first calculates the difference: (x+1,y) - (x,y) and (x,y+1) - (x,y)
        tv_h = ((img[:, :, 1:, :] - img[:, :, :-1, :]).pow(2)).sum()
        tv_w = ((img[:, :, :, 1:] - img[:, :, :, :-1]).pow(2)).sum()
        
        # Return the total variation loss
        return weight * (tv_h + tv_w)


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = 0
        
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()
        self.loss = 0
    
    @staticmethod
    def gram_matrix(input):
        batch_size, feature_maps, h, w = input.size()
        features = input.view(batch_size * feature_maps, h * w)
        G = torch.mm(features, features.t())
        return G.div(batch_size * feature_maps * h * w)
    
    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input
