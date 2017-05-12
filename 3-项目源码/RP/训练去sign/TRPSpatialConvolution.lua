local THNN = require 'nn.THNN'
local TRPSpatialConvolution, parent = torch.class('nn.TRPSpatialConvolution', 'nn.Module')

function TRPSpatialConvolution:__init(nInputPlane, nOutputPlane, compressionSize, kW, kH, dW, dH, padW, padH)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH

   self.dW = dW
   self.dH = dH
   self.padW = padW or 0
   self.padH = padH or self.padW

   self.weight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.gradBias = torch.Tensor(nOutputPlane)
   
   --
   self.tprojection = torch.randn(nInputPlane * kH * kW, compressionSize)
   self.gradTprojection = torch.Tensor(nInputPlane * kH * kW, compressionSize)
--   self.Biweight = torch.Tensor(nOutputPlane, compressionSize) 
   --
   
   self:reset()
end

function TRPSpatialConvolution:noBias()
   self.bias = nil
   self.gradBias = nil
   return self
end

function TRPSpatialConvolution:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
   end
   if nn.oldSeed then
      self.weight:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
      if self.bias then
         self.bias:apply(function()
         return torch.uniform(-stdv, stdv)
         end)
      end
   else
      self.weight:uniform(-stdv, stdv)
      if self.bias then
         self.bias:uniform(-stdv, stdv)
      end
   end
end

local function backCompatibility(self)
   self.finput = self.finput or self.weight.new()
   self.fgradInput = self.fgradInput or self.weight.new()
   if self.padding then
      self.padW = self.padding
      self.padH = self.padding
      self.padding = nil
   else
      self.padW = self.padW or 0
      self.padH = self.padH or 0
   end
   if self.weight:dim() == 2 then
      self.weight = self.weight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
   end
   if self.gradWeight and self.gradWeight:dim() == 2 then
      self.gradWeight = self.gradWeight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
   end
end

function TRPSpatialConvolution:updateOutput(input)
   assert(input.THNN, torch.type(input)..'.THNN backend not imported')
   backCompatibility(self)
   input.THNN.TRPSpatialConvolutionMM_updateOutput(
      input:cdata(),
      self.output:cdata(),
      self.weight:cdata(),
      THNN.optionalTensor(self.bias),
      self.finput:cdata(),
      self.fgradInput:cdata(),
      --
      self.tprojection:cdata(),
      --
      self.kW, self.kH,
      self.dW, self.dH,
      self.padW, self.padH
   )
   return self.output
end

function TRPSpatialConvolution:updateGradInput(input, gradOutput)
   assert(input.THNN, torch.type(input)..'.THNN backend not imported')
   if self.gradInput then
      backCompatibility(self)
      input.THNN.TRPSpatialConvolutionMM_updateGradInput(
         input:cdata(),
         gradOutput:cdata(),
         self.gradInput:cdata(),
         self.weight:cdata(),
         self.finput:cdata(),
         self.fgradInput:cdata(),
         --
         self.tprojection:cdata(),
         --
         self.kW, self.kH,
         self.dW, self.dH,
         self.padW, self.padH
      )
      return self.gradInput
   end
end

function TRPSpatialConvolution:accGradParameters(input, gradOutput, scale)
   assert(input.THNN, torch.type(input)..'.THNN backend not imported')
   scale = scale or 1
   backCompatibility(self)
   input.THNN.TRPSpatialConvolutionMM_accGradParameters(
      input:cdata(),
      gradOutput:cdata(),
      self.gradWeight:cdata(),
      THNN.optionalTensor(self.gradBias),
      self.finput:cdata(),
      self.fgradInput:cdata(),
      --
      self.tprojection:cdata(),
      self.gradTprojection:cdata(),
      self.weight:cdata(),      
      --
      self.kW, self.kH,
      self.dW, self.dH,
      self.padW, self.padH,
      scale
   )
end

function TRPSpatialConvolution:type(type,tensorCache)
   self.finput = self.finput and torch.Tensor()
   self.fgradInput = self.fgradInput and torch.Tensor()
   return parent.type(self,type,tensorCache)
end

function TRPSpatialConvolution:__tostring__()
   local s = string.format('%s(%d -> %d, %dx%d', torch.type(self),
         self.nInputPlane, self.nOutputPlane, self.kW, self.kH)
   if self.dW ~= 1 or self.dH ~= 1 or self.padW ~= 0 or self.padH ~= 0 then
     s = s .. string.format(', %d,%d', self.dW, self.dH)
   end
   if (self.padW or self.padH) and (self.padW ~= 0 or self.padH ~= 0) then
     s = s .. ', ' .. self.padW .. ',' .. self.padH
   end
   if self.bias then
      return s .. ')'
   else
      return s .. ') without bias'
   end
end

function TRPSpatialConvolution:clearState()
   nn.utils.clear(self, 'finput', 'fgradInput', '_input', '_gradOutput')
   return parent.clearState(self)
end
