#include <cstdio>
__global__ void add(const float* a, const float* b, float* c, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
int main(){
    const int n = 1<<20; size_t bytes = n*sizeof(float);
    float *a,*b,*c; cudaMallocManaged(&a,bytes); cudaMallocManaged(&b,bytes); cudaMallocManaged(&c,bytes);
    for(int i=0;i<n;i++){a[i]=1.f;b[i]=2.f;}
    add<<<(n+255)/256,256>>>(a,b,c,n);
    cudaDeviceSynchronize();
    printf("c[0]=%f c[n-1]=%f (expect 3)\n", c[0], c[n-1]);
}
