#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/RPSpatialConvolutionMMHtanh.c"
#else

static inline void THNN_(RPSpatialConvolutionMMHtanh_shapeCheck)(
  THTensor *input, THTensor *gradOutput,
  THTensor *weight, THTensor *bias,
  int kH, int kW, int dH, int dW, int padH, int padW) {

  THArgCheck(kW > 0 && kH > 0, 9,
         "kernel size should be greater than zero, but got kH: %d kW: %d", kH, kW);
  THArgCheck(dW > 0 && dH > 0, 11,
       "stride should be greater than zero, but got dH: %d dW: %d", dH, dW);
  THNN_ARGCHECK(weight->nDimension == 2 || weight->nDimension == 4, 5, weight,
    "2D or 4D weight tensor expected, but got: %s");

  if (bias != NULL) {
    THNN_CHECK_DIM_SIZE(bias, 1, 0, weight->size[0]);
  }

  int ndim = input->nDimension;
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  THNN_ARGCHECK(ndim == 3 || ndim == 4, 2, input,
    "3D or 4D input tensor expected but got: %s");

  long nInputPlane  = weight->size[1] / (kH * kW);
  long inputHeight  = input->size[dimh];
  long inputWidth   = input->size[dimw];
  long nOutputPlane = weight->size[0];
  long outputHeight = (inputHeight + 2*padH - kH) / dH + 1;
  long outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;

  if (outputWidth < 1 || outputHeight < 1)
    THError("Given input size: (%d x %d x %d). "
      "Calculated output size: (%d x %d x %d). Output size is too small",
      nInputPlane,inputHeight,inputWidth,nOutputPlane,outputHeight,outputWidth);

  THNN_CHECK_DIM_SIZE(input, ndim, dimf, nInputPlane);

  if (gradOutput != NULL) {
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimf, nOutputPlane);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimh, outputHeight);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimw, outputWidth);
  }
}

static void THNN_(RPSpatialConvolutionMMHtanh_Htanh)(
          THTensor *r,
          THTensor *t)
{
  real *r_data = THTensor_(data)(r);
  real *t_data = THTensor_(data)(t);
  ptrdiff_t sz = THTensor_(nElement)(r);
  ptrdiff_t i;
  for(i=0; i<sz; ++i){
    if(t_data[i] >= 1 || t_data[i] <= -1){
      r_data = 0;
    }
  }
}

static void THNN_(RPSpatialConvolutionMMHtanh_updateOutput_frame)(
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *finput,
          //
          THTensor *projection,
          //
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          long nInputPlane,
          long inputWidth,
          long inputHeight,
          long nOutputPlane,
          long outputWidth,
          long outputHeight)
{
  long i;
  THTensor *output2d;

  THNN_(unfolded_copy)(finput, input, kW, kH, dW, dH, padW, padH,
           nInputPlane, inputWidth, inputHeight,
           outputWidth, outputHeight);

  output2d = THTensor_(newWithStorage2d)(output->storage, output->storageOffset,
                                         nOutputPlane, -1,
                                         outputHeight*outputWidth, -1);

  if (bias) {
    for(i = 0; i < nOutputPlane; i++)
        THVector_(fill)
    (output->storage->data + output->storageOffset + output->stride[0] * i,
     THTensor_(get1d)(bias, i), outputHeight*outputWidth);
  } else {
    THTensor_(zero)(output);
  }

  //
  THTensor *Biweight = THTensor_(newWithSize2d)(nOutputPlane, projection->size[1]);
  THTensor_(addmm)(Biweight, 0, Biweight, 1, weight, projection); 
  THTensor_(sign)(Biweight,Biweight);

  THTensor *tprojection = THTensor_(new)();
  THTensor_(transpose)(tprojection, projection, 0, 1);
  THTensor *Binput = THTensor_(newWithSize2d)(projection->size[1],outputHeight*outputWidth);
  THTensor_(addmm)(Binput, 0, Binput, 1, tprojection, finput);
  THTensor_(sign)(Binput,Binput);
  THTensor_(addmm)(output2d, 1, output2d, 1, Biweight, Binput);

  THTensor_(free)(tprojection);
  THTensor_(free)(Biweight);
  THTensor_(free)(Binput);                  
  //
  //THTensor_(addmm)(output2d, 1, output2d, 1, weight, finput);

  THTensor_(free)(output2d);
}

void THNN_(RPSpatialConvolutionMMHtanh_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *finput,
          THTensor *fgradInput,
          //
          THTensor *projection,
          //
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH)
{
  int freeWeight = 0;

  if (weight->nDimension == 4) {
    long s1 = weight->size[0];
    long s2 = weight->size[1] * weight->size[2] * weight->size[3];
    weight = THTensor_(newWithStorage2d)(weight->storage, weight->storageOffset,
           s1, -1, s2, -1);
    freeWeight = 1;
  }

  THNN_(RPSpatialConvolutionMMHtanh_shapeCheck)
    (input, NULL, weight, bias, kH, kW, dH, dW, padH, padW);

  input = THTensor_(newContiguous)(input);
  int ndim = input->nDimension;
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  long nInputPlane = input->size[dimf];
  long inputHeight  = input->size[dimh];
  long inputWidth   = input->size[dimw];
  long nOutputPlane = weight->size[0];
  long outputHeight = (inputHeight + 2*padH - kH) / dH + 1;
  long outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;

  if(input->nDimension == 3)
  {
    //Only modify this part
    THTensor_(resize2d)(finput, kW*kH*nInputPlane, outputHeight*outputWidth);
    THTensor_(resize3d)(output, nOutputPlane, outputHeight, outputWidth);

    THNN_(RPSpatialConvolutionMMHtanh_updateOutput_frame)
      (input, output, weight, bias, finput, projection,
       kW, kH, dW, dH, padW, padH,
       nInputPlane, inputWidth, inputHeight,
       nOutputPlane, outputWidth, outputHeight);
  }
  else
  {
    long T = input->size[0];
    long t;

    THTensor_(resize3d)(finput, T, kW*kH*nInputPlane, outputHeight*outputWidth);
    THTensor_(resize4d)(output, T, nOutputPlane, outputHeight, outputWidth);

#pragma omp parallel for private(t)
    for(t = 0; t < T; t++)
    {
      THTensor *input_t = THTensor_(newSelect)(input, 0, t);
      THTensor *output_t = THTensor_(newSelect)(output, 0, t);
      THTensor *finput_t = THTensor_(newSelect)(finput, 0, t);

      THNN_(RPSpatialConvolutionMMHtanh_updateOutput_frame)
  (input_t, output_t, weight, bias, finput_t, projection,
   kW, kH, dW, dH, padW, padH,
   nInputPlane, inputWidth, inputHeight,
   nOutputPlane, outputWidth, outputHeight);

      THTensor_(free)(input_t);
      THTensor_(free)(output_t);
      THTensor_(free)(finput_t);
    }
  }

  THTensor_(free)(input);
  if (freeWeight)
    THTensor_(free)(weight);
}

static void THNN_(RPSpatialConvolutionMMHtanh_updateGradInput_frame)(
          THTensor *gradInput,
          THTensor *gradOutput,
          THTensor *weight, //no longer weight:t()
          THTensor *fgradInput,
          //
          THTensor *projection,
          THTensor *finput,
          //
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH)
{
  THTensor *gradOutput2d = THTensor_(newWithStorage2d)
    (gradOutput->storage, gradOutput->storageOffset,
     gradOutput->size[0], -1,
     gradOutput->size[1]*gradOutput->size[2], -1);

//
  THTensor *Biweight = THTensor_(newWithSize2d)(gradOutput2d->size[0], projection->size[1]);
  THTensor_(addmm)(Biweight, 0, Biweight, 1, weight, projection); 
  THTensor_(sign)(Biweight,Biweight);
  THTensor *tBiweight = THTensor_(new)();
  THTensor_(transpose)(tBiweight, Biweight, 0, 1);

  THTensor *tprojection = THTensor_(new)();
  THTensor_(transpose)(tprojection, projection, 0, 1);
  THTensor *Pinput = THTensor_(newWithSize2d)(projection->size[1],gradOutput2d->size[1]);
  THTensor_(addmm)(Pinput, 0, Pinput, 1, tprojection, finput);

  THTensor *gradBinput = THTensor_(newWithSize2d)(projection->size[1], gradOutput2d->size[1]);
  THTensor_(addmm)(gradBinput, 0, gradBinput, 1, tBiweight, gradOutput2d);

  THNN_(RPSpatialConvolutionMMHtanh_Htanh)(gradBinput, Pinput);

  THTensor_(addmm)(fgradInput, 0, fgradInput, 1, projection, gradBinput);
  
  THTensor_(free)(Biweight);
  THTensor_(free)(tBiweight);
  THTensor_(free)(tprojection);
  THTensor_(free)(Pinput);
  THTensor_(free)(gradBinput);  
//shen mingzhu 

/*Original one 
  THTensor_(addmm)(fgradInput, 0, fgradInput, 1, weight, gradOutput2d);
*/

  THTensor_(free)(gradOutput2d);

  THTensor_(zero)(gradInput);

  THNN_(unfolded_acc)(fgradInput, gradInput, kW, kH, dW, dH,
          padW, padH,
          gradInput->size[0], gradInput->size[2], gradInput->size[1],
          gradOutput->size[2], gradOutput->size[1]);
}

void THNN_(RPSpatialConvolutionMMHtanh_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *finput,
          THTensor *fgradInput,
          //
          THTensor *projection,
          //
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH)
{
  int freeWeight = 0;

  if (weight->nDimension == 4) {
    long s1 = weight->size[0];
    long s2 = weight->size[1] * weight->size[2] * weight->size[3];
    weight = THTensor_(newWithStorage2d)(weight->storage, weight->storageOffset,
           s1, -1, s2, -1);
    freeWeight = 1;
  }

  THNN_(RPSpatialConvolutionMMHtanh_shapeCheck)
    (input, gradOutput, weight, NULL, kH, kW, dH, dW, padH, padW);

  input = THTensor_(newContiguous)(input);
  gradOutput = THTensor_(newContiguous)(gradOutput);

  THTensor_(resizeAs)(gradInput, input);
  THTensor_(resizeAs)(fgradInput, finput);

  // depending on the BLAS library, fgradInput (result tensor) might
  // be left uninitialized on zero alpha, which might lead to weird behavior
  // hence, to be safe, zero it
  THTensor_(zero)(fgradInput);
//  THTensor *tweight = THTensor_(new)();
//  THTensor_(transpose)(tweight, weight, 0, 1);
//  the original put the tweight into the frame function as weight
  if(input->nDimension == 3)
  {
    //Only modify this part
    //modify tweight to weight
    THNN_(RPSpatialConvolutionMMHtanh_updateGradInput_frame)(gradInput, gradOutput,
                  weight, fgradInput, projection, finput,
                  kW, kH, dW, dH, padW, padH);
  }
  else
  {
    long T = input->size[0];
    long t;

#pragma omp parallel for private(t)
    for(t = 0; t < T; t++)
    {
      THTensor *gradInput_t = THTensor_(newSelect)(gradInput, 0, t);
      THTensor *gradOutput_t = THTensor_(newSelect)(gradOutput, 0, t);
      THTensor *fgradInput_t = THTensor_(newSelect)(fgradInput, 0, t);
      //
      THTensor *finput_t = THTensor_(newSelect)(finput, 0, t);
      //

      THNN_(RPSpatialConvolutionMMHtanh_updateGradInput_frame)(gradInput_t, gradOutput_t,
              weight, fgradInput_t, projection, finput_t,
              kW, kH, dW, dH, padW, padH);

      THTensor_(free)(gradInput_t);
      THTensor_(free)(gradOutput_t);
      THTensor_(free)(fgradInput_t);
    }
  }

//  THTensor_(free)(tweight);
  THTensor_(free)(input);
  THTensor_(free)(gradOutput);
  if (freeWeight)
    THTensor_(free)(weight);
}

static void THNN_(RPSpatialConvolutionMMHtanh_accGradParameters_frame)(
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *finput,
//
          THTensor *projection,
          THTensor *weight,
//
          real scale)
{
  long i;
  THTensor *gradOutput2d = THTensor_(newWithStorage2d)
    (gradOutput->storage, gradOutput->storageOffset,
     gradOutput->size[0], -1,
     gradOutput->size[1]*gradOutput->size[2], -1);

  //
  int freeWeight = 0;
  if (weight->nDimension == 4) {
    long s1 = weight->size[0];
    long s2 = weight->size[1] * weight->size[2] * weight->size[3];
    weight = THTensor_(newWithStorage2d)(weight->storage, weight->storageOffset,
           s1, -1, s2, -1);
    freeWeight = 1;
  }

  THTensor *tprojection = THTensor_(new)();
  THTensor_(transpose)(tprojection, projection, 0, 1);
  THTensor *Binput = THTensor_(newWithSize2d)(projection->size[1],gradOutput2d->size[1]);
  THTensor_(addmm)(Binput, 0, Binput, 1, tprojection, finput);
  THTensor_(sign)(Binput,Binput);

  THTensor *gradBiweight = THTensor_(newWithSize2d)(gradOutput2d->size[0], projection->size[1]);
  THTensor *tBinput = THTensor_(new)();
  THTensor_(transpose)(tBinput, Binput, 0, 1);
  THTensor_(addmm)(gradBiweight, 0, gradBiweight, 1, gradOutput2d, tBinput);

  THTensor *Pweight = THTensor_(newWithSize2d)(gradOutput2d->size[0], projection->size[1]);
  THTensor_(addmm)(Pweight, 0, Pweight, 1, weight, projection); 

  THNN_(RPSpatialConvolutionMMHtanh_Htanh)(gradBiweight, Pweight);

  THTensor_(addmm)(gradWeight, 0, gradWeight, 1, gradBiweight, tprojection);
  THTensor_(free)(tprojection);
  THTensor_(free)(Binput);
  THTensor_(free)(tBinput);
  THTensor_(free)(gradBiweight);
  THTensor_(free)(Pweight);
  //Shen Mingzhu
//  Original one 
//  THTensor_(transpose)(finput, finput, 0, 1);
//  THTensor_(addmm)(gradWeight, 1, gradWeight, scale, gradOutput2d, finput);
//  THTensor_(transpose)(finput, finput, 0, 1);
//
  if (gradBias) {
    for(i = 0; i < gradBias->size[0]; i++)
    {
      long k;
      real sum = 0;
      real *data = gradOutput2d->storage->data + gradOutput2d->storageOffset + i*gradOutput2d->stride[0];
      for(k = 0; k < gradOutput2d->size[1]; k++)
        sum += data[k];
      (gradBias->storage->data + gradBias->storageOffset)[i] += scale*sum;
    }
  }

  THTensor_(free)(gradOutput2d);
//
  if(freeWeight)
    THTensor_(free)(weight);
//Shen Mingzhu
}

void THNN_(RPSpatialConvolutionMMHtanh_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *finput,
          THTensor *fgradInput,
//
          THTensor *projection,
          THTensor *weight,
//
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          real scale)
{
  int freeWeight = 0;

  if (gradWeight->nDimension == 4) {
    long s1 = gradWeight->size[0];
    long s2 = gradWeight->size[1] * gradWeight->size[2] * gradWeight->size[3];
    gradWeight = THTensor_(newWithStorage2d)(gradWeight->storage,
               gradWeight->storageOffset,
               s1, -1, s2, -1);
    freeWeight = 1;
  }

  THNN_(RPSpatialConvolutionMMHtanh_shapeCheck)
    (input, gradOutput, gradWeight, gradBias, kH, kW, dH, dW, padH, padW);

  if(input->nDimension == 3)
  {
    //
    THNN_(RPSpatialConvolutionMMHtanh_accGradParameters_frame)(gradOutput, gradWeight,
              gradBias, finput, projection, weight, scale);
    //  
}
  else
  {
    long T = input->size[0];
    long t;

    for(t = 0; t < T; t++)
    {
      THTensor *gradOutput_t = THTensor_(newSelect)(gradOutput, 0, t);
      THTensor *finput_t = THTensor_(newSelect)(finput, 0, t);

      //
      THNN_(RPSpatialConvolutionMMHtanh_accGradParameters_frame)(gradOutput_t, gradWeight,
                gradBias, finput_t, projection, weight, scale);
      //

      THTensor_(free)(gradOutput_t);
      THTensor_(free)(finput_t);
    }
  }
  if (freeWeight)
    THTensor_(free)(gradWeight);
}

#endif

