// Bit-identity harness for hc_mix.cu against a transcription of the exact
// candle-kernels each fused op replaces. Poison = 1-ULP single-element.
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

extern "C" int hc_mix_params_f32(const void*,const void*,const void*,const void*,void*,void*,void*,int,int,float,long long);
extern "C" int hc_y_combine(const void*,const void*,void*,int,int,int,int,long long);

// ---- reference: the candle op chain, one kernel per op, exactly as candle runs it ----
__global__ void k_bmul(const float*a,const float*b,float*o,int n,int stride_b){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)o[i]=a[i]*b[i/stride_b];}
__global__ void k_bmul_s(const float*a,const float*s,float*o,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)o[i]=a[i]*s[0];}
__global__ void k_badd(const float*a,const float*b,float*o,int n,int bn){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)o[i]=a[i]+b[i%bn];}
__device__ __forceinline__ float recipg(float a){return 1.0/a;}   // candle: DOUBLE literal
__global__ void k_sigmoid(const float*a,float*o,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)o[i]=recipg(1.0f+expf(-a[i]));}
__global__ void k_affine(const float*a,float*o,int n,float mul,float add){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)o[i]=a[i]*mul+add;}
// candle fast_sum over dim1 of [n,HC,h]: identity-padded pairwise tree
__global__ void k_sum1(const float*x,float*o,int n,int HC,int h){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n*h) return;
  int row=i/h, col=i-row*h; float b[16];
  for(int t=0;t<HC;++t) b[t]=0.0f+x[(size_t)row*HC*h+(size_t)t*h+col];
  for(int s=HC>>1;s>0;s>>=1) for(int t=0;t<s;++t) b[t]=b[t]+b[t+s];
  o[i]=b[0];
}
__global__ void k_cast_bf16(const float*a,__nv_bfloat16*o,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)o[i]=static_cast<__nv_bfloat16>(a[i]);}

static void ck(const char*w){cudaError_t e=cudaGetLastError();if(e!=cudaSuccess){fprintf(stderr,"CUDA %s: %s\n",w,cudaGetErrorString(e));exit(2);}}

int main(int argc,char**argv){
  const int n=argc>1?atoi(argv[1]):3, HC=4, h=argc>2?atoi(argv[2]):4096;
  const int poison=argc>3?atoi(argv[3]):0;
  const int MIX=(2+HC)*HC; const float eps=1e-6f;
  std::vector<float> hmix(n*MIX),hrs(n),hsc(3),hb(MIX),hx(n*HC*h);
  for(int i=0;i<n*MIX;++i)hmix[i]=0.7f*sinf(0.31f*i)+0.05f*(i%5);
  for(int i=0;i<n;++i)hrs[i]=0.4f+0.13f*i;
  hsc[0]=0.83f;hsc[1]=1.27f;hsc[2]=0.61f;
  for(int i=0;i<MIX;++i)hb[i]=0.2f*cosf(0.47f*i);
  for(size_t i=0;i<hx.size();++i)hx[i]=0.5f*sinf(0.017f*(float)i)+0.01f*(float)(i%11);
  // (input poison removed: a shared input perturbs BOTH paths identically)

  float *dmix,*drs,*dsc,*db,*dpre,*dpost,*dcomb,*dx,*dpre_r,*dpost_r,*dcomb_r,*dtmp,*dtmp2,*dyf;
  __nv_bfloat16 *dy,*dy_r;
  cudaMalloc(&dmix,n*MIX*4);cudaMalloc(&drs,n*4);cudaMalloc(&dsc,12);cudaMalloc(&db,MIX*4);
  cudaMalloc(&dpre,n*HC*4);cudaMalloc(&dpost,n*HC*4);cudaMalloc(&dcomb,n*HC*HC*4);
  cudaMalloc(&dpre_r,n*HC*4);cudaMalloc(&dpost_r,n*HC*4);cudaMalloc(&dcomb_r,n*HC*HC*4);
  cudaMalloc(&dx,(size_t)n*HC*h*4);cudaMalloc(&dtmp,n*MIX*4);cudaMalloc(&dtmp2,n*MIX*4);
  cudaMalloc(&dyf,(size_t)n*h*4);cudaMalloc(&dy,(size_t)n*h*2);cudaMalloc(&dy_r,(size_t)n*h*2);
  cudaMemcpy(dmix,hmix.data(),n*MIX*4,cudaMemcpyHostToDevice);
  cudaMemcpy(drs,hrs.data(),n*4,cudaMemcpyHostToDevice);
  cudaMemcpy(dsc,hsc.data(),12,cudaMemcpyHostToDevice);
  cudaMemcpy(db,hb.data(),MIX*4,cudaMemcpyHostToDevice);
  cudaMemcpy(dx,hx.data(),(size_t)n*HC*h*4,cudaMemcpyHostToDevice);

  // ---- fused ----
  if(!hc_mix_params_f32(dmix,drs,dsc,db,dpre,dpost,dcomb,n,HC,eps,0)){printf("params fallback\n");return 2;}
  if(!hc_y_combine(dx,dpre,dy,n,HC,h,1,0)){printf("y fallback\n");return 2;}
  cudaDeviceSynchronize();ck("fused");

  // ---- reference chain ----
  int B=256, G=(n*MIX+B-1)/B;
  k_bmul<<<G,B>>>(dmix,drs,dtmp,n*MIX,MIX);                     // mixes = raw*rsqrt
  // pre
  k_bmul_s<<<G,B>>>(dtmp,dsc+0,dtmp2,n*MIX);
  k_badd<<<G,B>>>(dtmp2,db,dtmp2,n*MIX,MIX);
  k_sigmoid<<<G,B>>>(dtmp2,dtmp2,n*MIX);
  k_affine<<<G,B>>>(dtmp2,dtmp2,n*MIX,1.0f,eps);
  for(int r=0;r<n;++r) cudaMemcpy(dpre_r+r*HC,dtmp2+r*MIX,HC*4,cudaMemcpyDeviceToDevice);
  // post
  k_bmul_s<<<G,B>>>(dtmp,dsc+1,dtmp2,n*MIX);
  k_badd<<<G,B>>>(dtmp2,db,dtmp2,n*MIX,MIX);
  k_sigmoid<<<G,B>>>(dtmp2,dtmp2,n*MIX);
  k_affine<<<G,B>>>(dtmp2,dtmp2,n*MIX,2.0f,0.0f);
  for(int r=0;r<n;++r) cudaMemcpy(dpost_r+r*HC,dtmp2+r*MIX+HC,HC*4,cudaMemcpyDeviceToDevice);
  // comb
  k_bmul_s<<<G,B>>>(dtmp,dsc+2,dtmp2,n*MIX);
  k_badd<<<G,B>>>(dtmp2,db,dtmp2,n*MIX,MIX);
  for(int r=0;r<n;++r) cudaMemcpy(dcomb_r+r*HC*HC,dtmp2+r*MIX+2*HC,HC*HC*4,cudaMemcpyDeviceToDevice);
  // y tail: bmul over [n,HC,h] then sum(1) then cast  (uses REFERENCE pre)
  {
    int T=n*h, G2=(T+B-1)/B;
    k_sum1<<<G2,B>>>(dx,dyf,n,HC,h);  // placeholder shape; real op below
    // proper: multiply first
    float* dprod; cudaMalloc(&dprod,(size_t)n*HC*h*4);
    int T3=n*HC*h, G3=(T3+B-1)/B;
    // pre broadcast over h within each (row,i)
    k_bmul<<<G3,B>>>(dx,dpre_r,dprod,T3,h);   // stride h -> index (row*HC+i)
    k_sum1<<<G2,B>>>(dprod,dyf,n,HC,h);
    k_cast_bf16<<<G2,B>>>(dyf,dy_r,T);
    cudaFree(dprod);
  }
  cudaDeviceSynchronize();ck("reference");

  // NEGATIVE CONTROL: perturb the FUSED result by exactly 1 ULP at a known index.
  // This cannot cancel and cannot be absorbed by the reference path.
  if(poison){
    unsigned u; float f;
    cudaMemcpy(&f,dpre,4,cudaMemcpyDeviceToHost); memcpy(&u,&f,4); u+=1u; memcpy(&f,&u,4);
    cudaMemcpy(dpre,&f,4,cudaMemcpyHostToDevice);
    unsigned short h16; cudaMemcpy(&h16,dy,2,cudaMemcpyDeviceToHost); h16+=1;
    cudaMemcpy(dy,&h16,2,cudaMemcpyHostToDevice);
    unsigned short c16; cudaMemcpy(&c16,dcomb,2,cudaMemcpyDeviceToHost); c16+=1;
    cudaMemcpy(dcomb,&c16,2,cudaMemcpyHostToDevice);
    cudaMemcpy(&f,dpost,4,cudaMemcpyDeviceToHost); memcpy(&u,&f,4); u+=1u; memcpy(&f,&u,4);
    cudaMemcpy(dpost,&f,4,cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
  }

  auto cmp=[&](const char*name,const void*a,const void*b,size_t bytes){
    std::vector<unsigned char> A(bytes),Bv(bytes);
    cudaMemcpy(A.data(),a,bytes,cudaMemcpyDeviceToHost);
    cudaMemcpy(Bv.data(),b,bytes,cudaMemcpyDeviceToHost);
    int ok=memcmp(A.data(),Bv.data(),bytes)==0;
    size_t diff=0; for(size_t i=0;i<bytes;++i) if(A[i]!=Bv[i]) diff++;
    printf("  %-10s %s  (%zu/%zu bytes differ)\n",name,ok?"BIT-IDENTICAL":"*** MISMATCH ***",diff,bytes);
    return ok;
  };
  printf("n=%d hc=%d h=%d poison=%d\n",n,HC,h,poison);
  int ok=1;
  ok &= cmp("pre",dpre,dpre_r,(size_t)n*HC*4);
  ok &= cmp("post",dpost,dpost_r,(size_t)n*HC*4);
  ok &= cmp("comb",dcomb,dcomb_r,(size_t)n*HC*HC*4);
  ok &= cmp("y(bf16)",dy,dy_r,(size_t)n*h*2);
  // fingerprint so poison provably changes the output
  std::vector<unsigned> fp(4); cudaMemcpy(fp.data(),dpre,16,cudaMemcpyDeviceToHost);
  printf("  FP pre[0..3] %08x %08x %08x %08x\n",fp[0],fp[1],fp[2],fp[3]);
  printf("%s\n", ok?"ALL BIT-IDENTICAL":"FAILED");
  return ok?0:1;
}
