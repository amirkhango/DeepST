1. Install [**DeepST**](https://github.com/lucktroy/DeepST). 

2. Download [**TaxiBJ**](https://github.com/lucktroy/DeepST/tree/master/data/TaxiBJ) data

3. Reproduce the results of ST-ResNet and its variants. 

    * Result of Model **L2-E**

    ```
    THEANO_FLAGS="device=gpu,floatX=float32" python exptTaxiBJ.py 2
    ```

    * Result of Model **L4-E**

    ```
    THEANO_FLAGS="device=gpu,floatX=float32" python exptTaxiBJ.py 4
    ```
    
    * Result of Model **L12-E**

    ```
    THEANO_FLAGS="device=gpu,floatX=float32" python exptTaxiBJ.py 12
    ```


    * Result of Model **L12**
    ```
    THEANO_FLAGS="device=gpu,floatX=float32" python exptTaxiBJ-L12.py
    ```