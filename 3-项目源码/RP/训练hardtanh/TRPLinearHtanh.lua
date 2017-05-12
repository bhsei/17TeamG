local TRPLinearHtanh, parent = torch.class('nn.TRPLinearHtanh', 'nn.Module')

--add a hyperparameter compressionSize which is used to be the size of compressed input
function TRPLinearHtanh:__init(inputSize, outputSize, compressionSize, bias)
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

function TRPLinearHtanh:noBias()
   self.bias = nil
   self.gradBias = nil
   return self
end

function TRPLinearHtanh:reset(stdv)
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
function TRPLinearHtanh:updateOutput(input)
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
function TRPLinearHtanh:updateGradInput(input, gradOutput)
   if self.gradInput then
      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
         --self.gradInput:addmv(0, 1, self.weight:t(), gradOutput)
         local gradBinput = torch.Tensor(self.tprojection:size(2))
         gradBinput:addmv(0, 1, self.Biweight:t(), gradOutput)
         local Pinput = torch.mv(self.tprojection:t(), input)
         gradBinput[Pinput:ge(1)]=0
         gradBinput[Pinput:le(-1)]=0
         self.gradInput:addmv(0, 1, self.tprojection, gradBinput)
         --mingzhu
      elseif input:dim() == 2 then
         --self.gradInput:addmm(0, 1, gradOutput, self.weight)
         local gradBinput = torch.mm(gradOutput, self.Biweight)
         local Pinput = torch.mm(input, self.tprojection)
         gradBinput[Pinput:ge(1)]=0
         gradBinput[Pinput:le(-1)]=0
         self.gradInput:addmm(0, 1, gradBinput, self.tprojection:t())
         --shen mingzhu
      end

      return self.gradInput
   end
end

--we only modify the part when input:dim() equals 1
function TRPLinearHtanh:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      --self.gradWeight:addr(scale, gradOutput, input)
      local gradBiweight = torch.Tensor(self.weight:size(1), self.tprojection:size(2))
      gradBiweight:addr(0, 1, gradOutput, Binput)
      local Pweight = torch.mm(self.weight, self.tprojection)
      gradBiweight[Pweight:ge(1)]=0
      gradBiweight[Pweight:le(-1)]=0 
      self.gradWeight:addmm(scale, gradBiweight, self.tprojection:t())
      --gradWeight gradient

      local gradBinput = torch.Tensor(self.tprojection:size(2))
      gradBinput:addmv(0, 1, self.Biweight:t(), gradOutput)
      local Pinput = torch.mv(self.tprojection:t(), input)
      gradBinput[Pinput:ge(1)]=0
      gradBinput[Pinput:le(-1)]=0

      local delta = torch.mm(self.weight:t(), gradBiweight)
      delta:addr(1, 1, input, gradBinput)
      --gradtProjection = weight:t()*gradOutput*Binput+input*gradOutput:t()*Biweight
      self.gradTprojection:add(scale, delta)
      --gradTprojection gradient
      --mingzhu 
      if self.bias then self.gradBias:add(scale, gradOutput) end
   elseif input:dim() == 2 then
      --self.gradWeight:addmm(scale, gradOutput:t(), input)
      local gradBiweight = torch.mm(gradOutput:t(), self.Binput)
      local Pweight = torch.mm(self.weight, self.tprojection)
      gradBiweight[Pweight:ge(1)]=0
      gradBiweight[Pweight:le(-1)]=0
      self.gradWeight:addmm(scale, gradBiweight, self.tprojection:t())      --gradWeight gradient
      --gradWeight gradient

      local gradBinput = torch.mm(gradOutput, self.Biweight)
      local Pinput = torch.mm(input, self.tprojection)
      gradBinput[Pinput:ge(1)]=0
      gradBinput[Pinput:le(-1)]=0

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

function TRPLinearHtanh:sharedAccUpdateGradParameters(input, gradOutput, lr)
   -- we do not need to accumulate parameters when sharing:
   self:defaultAccUpdateGradParameters(input, gradOutput, lr)
end

function TRPLinearHtanh:clearState()
   if self.addBuffer then self.addBuffer:set() end
   return parent.clearState(self)
end

function TRPLinearHtanh:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1)) ..
      (self.bias == nil and ' without bias' or '')
end
