# PGFS - Unsupervised Feature Selection via Principal Graph Learning
A pipeline to characterize the underlying manifold of biology data and select relevant features.

## Setup
The setup process for the proposed pipeline requires the following steps:
### Download the pipeline
```bash
git clone https://github.com/liluacrobat/MicroDynamics.git
```

### Requirement
The following software is required
* MATLAB

## Usage
### Input
```
DATA: data matrix whose columns are samples and rows are features
```
### Optional arguments  
```
Para: parameters used by PGFS
    sigma - kernel width (default:0.01) 
    beta - tree length regularization (default:10)
    lambda - sparness regularization (default:16)
    it - maximum iteration (defayult:50)
    M - number of points on the principal graph (default: number of samples)
```    
### Output
```
Weight: feature weight
Y: latent points on the principal graph 
B: edge indicators
R: soft assignment coefficients
Objective: objective value
History: records of the leasrning process
MSE: mean squared error
TreeLength: tree length
```
## Example
We provide a demo of applying PGFS to a prostate cancer data set [1] to demonstrate its utility for solving a real-world problem. The experimental data (accession number GSE6919) is downloaded from NCBI's Gene Expression Omnibus (GEO). The data set contains gene expression information of over 12,000 genes obtained from 168 patient tissue samples, including 17 from normal prostate (NP), 60 normal tissues adjacent to tumors (NA), 66 primary tumors (PT), and 25 metastatic tumors (MT), which can be regarded as a progressive series of prostate disease. 

We can perform feature selection while uncovering the cancer progression structure embedded in the high-dimentional space by running 'DEMO_PGFS.m' under the 'example' directory. The precalculated results is stored in the 'example/precalculated/'. 

## Reference
[1] U. R. Chandran, C. Ma, R. Dhir, M. Bisceglia, M. Lyons-Weiler, W. Liang, G. Michalopoulos, M. Becich, and F. A. Monzon, “Gene
expression profiles of prostate cancer reveal involvement of multiple molecular pathways in the metastatic process,” BMC Cancer, vol. 7, no. 1, p. 64, 2007.
