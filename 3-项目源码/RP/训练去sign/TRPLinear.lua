local TRPLinear, parent = torch.class('nn.TRPLinear', 'nn.Module')

--add a hyperparameter compressionSize which is used to be the size of compressed input
function TRPLinear:__init(inputSize, outputSize, compressionSize, bias)
--mingzhu   
   parent.__init(self)
   local bias = ((bias == nil) and true) or bias
   self.weight = torch.Tensor(outputSize, inputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   if bias then
      self.bias = torch.Tensor(outputSize)
      self.gradBias = torch.Tensor(outputSize)
   end
   --initialize tprojection matrix to a inputSize * compressionSize Tensor of random numbers from a normal distribution with mean zero and variance one
   self.tprojection = torch.randn(inputSize, compressionSize)
   self.gradTprojection = torch.Tensor(inputSize, compressionSize)
   self.Binput = torch.Tensor()
   self.Biweight = torch.Tensor()
   --mingzhu 
   self:reset()
end

function TRPLinear:noBias()
   self.bias = nil
   self.gradBias = nil
   return self
end

function TRPLinear:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
      end
      if self.bias then
         for i=1,self.bias:nElement() do
            self.bias[i] = torch.uniform(-stdv, stdv)
         end
      end
   else
      self.weight:uniform(-stdv, stdv)
      if self.bias then self.bias:uniform(-stdv, stdv) end
   end
   return self
end

local function updateAddBuffer(self, input)
   local nframe = input:size(1)
   self.addBuffer = self.addBuffer or input.new()
   if self.addBuffer:nElement() ~= nframe then
      self.addBuffer:resize(nframe):fill(1)
   end
end

--Binput, Biweight is the binary matrix of input and weight
--output = Biweight*Binput+bias
--we only modify the part when the input.dim()==1
function TRPLinear:updateOutput(input)
   if input:dim() == 1 then
      self.output:resize(self.weight:size(1))
      if self.bias then 
         self.output:copy(self.bias) 
      else 
         self.output:zero() 
      end
      --
      --self.Biweight = torch.Tensor(outputSize, compressionSize)
      --self.Binput = torch.Tensor(compressionSize)      
      self.Biweight= torch.sign(torch.mm(self.weight, self.tprojection))
      self.Binput = torch.sign(torch.mv(self.tprojection:t(), input))
      self.output:addmv(1, self.Biweight, self.Binput)
      --self.output:addmv(1, self.weight, input)
      --mingzhu
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nframe, self.weight:size(1))
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      updateAddBuffer(self, input)
      --
      --self.Biweight = torch.Tensor(outputSize, compressionSize)
      --self.Binput = torch.Tensor(input:size(1), compressionSize)
      self.Biweight= torch.sign(torch.mm(self.weight, self.tprojection))
      self.Binput = torch.sign(torch.mm(input, self.tprojection))
      self.output:addmm(0, self.output, 1, self.Binput, self.Biweight:t())
      --self.output:addmm(0, self.output, 1, input, self.weight:t())
      --shen mingzhu      
      if self.bias then self.output:addr(1, self.addBuffer, self.bias) end
   else
      error('input must be vector or matrix')
   end

   return self.output
end

--for the derivative of sign function, I choose the ignore it, 
function TRPLinear:updateGradInput(input, gradOutput)
   if self.gradInput then
      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
         --self.gradInput:addmv(0, 1, self.weight:t(), gradOutput)
         self.gradInput:addmv(0, 1, self.tprojection*self.Biweight:t(), gradOutput)
         --mingzhu
      elseif input:dim() == 2 then
         --self.gradInput:addmm(0, 1, gradOutput, self.weight)
         self.gradInput:addmm(0, 1, gradOutput, torch.mm(self.Biweight, self.tprojection:t()))
         --shen mingzhu
      end

      return self.gradInput
   end
end

--we only modify the part when input:dim() equals 1
function TRPLinear:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      --self.gradWeight:addr(scale, gradOutput, input)
      self.gradWeight:addr(scale, gradOutput, torch.mv(self.tprojection, self.Binput))
      --gradWeight gradient

      local gradBiweight = torch.Tensor(self.weight:size(1), self.tprojection:size(2))
      gradBiweight:addr(0, 1, gradOutput, Binput)

      local gradBinput = torch.Tensor(self.tprojection:size(2))
      gradBinput:addmv(0, 1, self.Biweight:t(), gradOutput)

      local delta = torch.mm(self.weight:t(), gradBiweight)
      delta:addr(1, 1, input, gradBinput)
      --gradtProjection = weight:t()*gradOutput*Binput+input*gradOutput:t()*Biweight
      self.gradTprojection:add(scale, delta)
      --gradTprojection gradient
      --mingzhu 
      if self.bias then self.gradBias:add(scale, gradOutput) end
   elseif input:dim() == 2 then
      --self.gradWeight:addmm(scale, gradOutput:t(), input)
      self.gradWeight:addmm(scale, gradOutput:t(), torch.mm(self.Binput, self.tprojection:t()))
      --gradWeight gradient

      local gradBiweight = torch.mm(gradOutput:t(), self.Binput)
      local gradBinput = torch.mm(gradOutput, self.Biweight)
      local delta = torch.mm(self.weight:t(), gradBiweight) + torch.mm(input:t(), gradBinput)
      self.gradTprojection:add(scale, delta)  
      --gradTprojection gradient    
      --shen mingzhu      
      if self.bias then
         -- update the size of addBuffer if the input is not the same size as the one we had in last updateGradInput
         updateAddBuffer(self, input)
         self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
      end
   end
end

function TRPLinear:sharedAccUpdateGradParameters(input, gradOutput, lr)
   -- we do not need to accumulate parameters when sharing:
   self:defaultAccUpdateGradParameters(input, gradOutput, lr)
end

function TRPLinear:clearState()
   if self.addBuffer then self.addBuffer:set() end
   return parent.clearState(self)
end

function TRPLinear:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1)) ..
      (self.bias == nil and ' without bias' or '')
end
