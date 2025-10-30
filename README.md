# CUDA Practice

I don't own a CUDA-equipped GPU, so I ran all of my tests on a small GPU-equpped AWS EC2 instance. I controlled this instance via SSH on my local machine's terminal (MacOS).

### Contents

1. [Setup](#0-setup)
1. [Hello World](#1-hello-world)
2. [Vector Add](#2-vector-add)
3. [Image Processor](#3-image-processor)
4. [Flash Attention](#4-flash-attention)

### 0. Setup

To Launch an EC2 Instance, navigate to the EC2 instances page on the AWS console and click `Launch Instances`. Use these configuration options:
1. Amazon Machine Image (AMI): `Deep Learning Base AMI with Single CUDA`
2. Instance Type: `g4dn.xlarge` (Cheapest Available GPU Instance on AWS; Costs $0.53/hr)

Leave everything else as their defaults or modify as desired. 


**IMPORTANT:** By default, AWS doesn't allow you to launch GPU instances; you need to request a limit increase in `Service Quotas` (another AWS service). Go to Dashboard -> Amazon Elastic Compute Cloud (Amazon EC2) -> Running On-Demand G and VT instances -> Request increase at account level -> Enter a quota value (I requested 4 vCPUs). My request got approved within 2 hours.

**IMPORTANT:** Make sure to terminate/delete your EC2 instance when you're done; AWS will still charge you if the instance is active, even if you're not actively running a program. Make sure to verify in the console that your instance was deleted when you're done.


Once your quota request is approved and you successfully launch your instance, take note of the public IPv4 address. You can SSH into your instance from your local terminal (NOTE: My local machine is a Mac):

`ssh -i ~/path/to/your/key/pair/file.pem ec2-user@[your instance IPv4 address]`

I used `scp` to put my local files in my EC2 instance.

General `scp` Syntax: `scp -i ~/path/to/your/key/pair/file.pem ~/path/to/your/cuda/file.cu ec2-user@[EC2 Instance IPv4 address]:~/ `

### 1. Hello World

I created this demo to familiarize myself with CUDA. 

1. `scp -i ~/path/to/your/key/pair/file.pem ~/cuda_practice/hello_world/hello.cu ec2-user@[EC2 IPv4 address]:~/`
2. `ssh -i ~/path/to/your/key/pair/file.pem ec2-user@[EC2 IPv4 address]`
3. (Inside EC2 Instance) `nvcc hello.cu -o hello`
4. `./hello`

### 2. Vector Add

I created a CUDA program to test its speed on a parallelized task: adding two vectors.

Vector addition is element-wise and each element's computation is independent, which makes vector addition very GPU-friendly; instead of a for-loop approach (slow for large vectors), we can create a thread for each element.

To compare speed, I first wrote a simple for-loop program to add two random vectors in Python and C++ and lastly in CUDA – while timing everything. Using Python as the baseline, C++ was ~10x faster and CUDA was ~1000x faster.


1. `scp -i ~/path/to/your/key/pair/file.pem ~/cuda_practice/vector_add/vector_add.cu ec2-user@[EC2 IPv4 address]:~/`
2. `scp -i ~/path/to/your/key/pair/file.pem ~/cuda_practice/vector_add/vector_add.py ec2-user@[EC2 IPv4 address]:~/`
3. `ssh -i ~/path/to/your/key/pair/file.pem ec2-user@[EC2 IPv4 address]`
4. (Inside your EC2 Instance) `nvcc vector_add.cu -o vec_add`
5. `./vec_add`
6. Run `./vec_add` a second time and observe that the CUDA portion runs about twice as fast as the first runtime.

### 3. Image Processor


### 4. Flash Attention