local RPLinearHtanh, parent = torch.class('nn.RPLinearHtanh', 'nn.Module')

--add a hyperparameter compressionSize which is used to be the size of compressed input
function RPLinearHtanh:__init(inputSize, outputSize, compressionSize, bias)
--test
   parent.__init(self)
   local bias = ((bias == nil) and true) or bias
   self.weight = torch.Tensor(outputSize, inputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   if bias then
      self.bias = torch.Tensor(outputSize)
      self.gradBias = torch.Tensor(outputSize)
   end
   --initialize projection matrix to a inputSize * compressionSize Tensor of random numbers from a normal distribution with mean zero and variance one
   self.projection = torch.randn(inputSize, compressionSize)
   self.Binput = torch.Tensor()
   self.Biweight = torch.Tensor()
   --test    
   self:reset()
end

function RPLinearHtanh:noBias()
   self.bias = nil
   self.gradBias = nil
   return self
end

function RPLinearHtanh:reset(stdv)
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

function RPLinearHtanh:updateOutput(input)
   if input:dim() == 1 then
      self.output:resize(self.weight:size(1))
      if self.bias then self.output:copy(self.bias) else self.output:zero() end
      --
      --self.Biweight = torch.Tensor(outputSize, compressionSize)
      --self.Binput = torch.Tensor(compressionSize)
      self.Biweight= torch.sign(torch.mm(self.weight, self.projection))
      self.Binput = torch.sign(torch.mv(self.projection:t(), input))
      self.output:addmv(1, self.Biweight, self.Binput)
      --self.output:addmv(1, self.weight, input)
      --test
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
      self.Biweight= torch.sign(torch.mm(self.weight, self.projection))
      self.Binput = torch.sign(torch.mm(input, self.projection))
      self.output:addmm(0, self.output, 1, self.Binput, self.Biweight:t())
      --self.output:addmm(0, self.output, 1, input, self.weight:t())
      -- test      
      if self.bias then self.output:addr(1, self.addBuffer, self.bias) end
   else
      error('input must be vector or matrix')
   end

   return self.output
end

function RPLinearHtanh:updateGradInput(input, gradOutput)
   if self.gradInput then

      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
         --self.gradInput:addmv(0, 1, self.weight:t(), gradOutput)
         local gradBinput = torch.Tensor(self.projection:size(2))
         gradBinput:addmv(0, 1, self.Biweight:t(), gradOutput)
         local Pinput = torch.mv(self.projection:t(), input)
         gradBinput[Pinput:ge(1)]=0
         gradBinput[Pinput:le(-1)]=0
         self.gradInput:addmv(0, 1, self.projection, gradBinput)
         --test
      elseif input:dim() == 2 then
         --self.gradInput:addmm(0, 1, gradOutput, self.weight)
         local gradBinput = torch.mm(gradOutput, self.Biweight)
         local Pinput = torch.mm(input, self.projection)
         gradBinput[Pinput:ge(1)]=0
         gradBinput[Pinput:le(-1)]=0
         self.gradInput:addmm(0, 1, gradBinput, self.projection:t())
         -- test
      end

      return self.gradInput
   end
end

function RPLinearHtanh:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      --self.gradWeight:addr(scale, gradOutput, input)
      local gradBiweight = torch.Tensor(self.weight:size(1), self.projection:size(2))
      gradBiweight:addr(0, 1, gradOutput, Binput)
      local Pweight = torch.mm(self.weight, self.projection)
      gradBiweight[Pweight:ge(1)]=0
      gradBiweight[Pweight:le(-1)]=0 
      self.gradWeight:addmm(scale, gradBiweight, self.projection:t())
      --test 
      if self.bias then self.gradBias:add(scale, gradOutput) end
   elseif input:dim() == 2 then      
      --self.gradWeight:addmm(scale, gradOutput:t(), input)
      local gradBiweight = torch.mm(gradOutput:t(), self.Binput)
      local Pweight = torch.mm(self.weight, self.projection)
      gradBiweight[Pweight:ge(1)]=0
      gradBiweight[Pweight:le(-1)]=0
      self.gradWeight:addmm(scale, gradBiweight, self.projection:t())
      -- test
      if self.bias then
         -- update the size of addBuffer if the input is not the same size as the one we had in last updateGradInput
         updateAddBuffer(self, input)
         self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
      end
   end
end

-- we do not need to accumulate parameters when sharing
RPLinearHtanh.sharedAccUpdateGradParameters = RPLinearHtanh.accUpdateGradParameters

function RPLinearHtanh:clearState()
   if self.addBuffer then self.addBuffer:set() end
   return parent.clearState(self)
end

function RPLinearHtanh:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1)) ..
      (self.bias == nil and ' without bias' or '')
end