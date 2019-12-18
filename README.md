<img src="figs/PEAKurban.png" alt="PEAK Urban logo" align="right" width ="200" height="133">

<img src="figs/logo_rise_eafit.png" alt="RiSE-group logo" align="middle" width ="380" height="100">


Spatiotemporal modeling of urban growth using machine learning
=================================================================


## Description

 This repository contains all the input variables and the autotuning program that were used in the paper "Spatiotemporal modeling of urban growth using machine learning", by Jairo A. Gómez, Jorge E. Patiño, Juan C. Duque, and Santiago Passos.

#### "Spatiotemporal modeling of urban growth using machine learning"

Jairo A. Gómez<sup>1</sup>, Jorge E. Patiño<sup>1</sup>, Juan C. Duque<sup>1</sup>, Santiago Passos<sup>1</sup>

<sup>1</sup> RiSE-group, Department of Mathematical Sciences, Universidad EAFIT, Medellin, Colombia



maintainer = "RiSE Group" (http://www.rise-group.org/). Universidad EAFIT

Corresponding author = jagomeze at eafit.edu.co; Tel.: +57-4-2619500 EXT 9894.

### Abstract 

This paper presents a general framework for modeling the growth of three important variables for cities: population distribution, binary urban footprint, and urban footprint in color. The framework models the population distribution as a spatiotemporal regression problem using machine learning, and it obtains the binary urban footprint from the population distribution through a binary classifier plus a temporal correction for existing urban regions. The framework estimates the urban footprint in color from its previous value, as well as from past and current values of the binary urban footprint using a semantic inpainting algorithm. By combining this framework with free data from the Landsat archive and the Global Human Settlement Layer framework, interested users can get approximate growth predictions of any city in the world. These predictions can be improved with the inclusion in the framework of additional spatially distributed input variables over time subject to availability. Unlike widely used growth models based on cellular automata, there are two main advantages of using the proposed machine learning-based framework. Firstly, it does not require to define rules a priori because the model learns the dynamics of growth directly from the historical data. Secondly, it is very easy to train new machine learning models using different explanatory input variables to assess their impact. As a proof of concept, we tested the framework in Valledupar and Rionegro, two Latin American cities located in Colombia with different geomorphological characteristics, and found that the model predictions were in close agreement with the ground-truth based on performance metrics such as the root-mean-square error, zero-mean normalized cross-correlation, Pearson's correlation coefficient for continuous variables, and a few others for discrete variables such as the intersection over union, accuracy, and f1 metric. In summary, our framework for modeling urban growth is flexible, allows sensitivity analyses, and can help policymakers worldwide to assess different what-if scenarios during the planning cycle of sustainable and resilient cities.



## License

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

## Bibtext entry

```tex
@article{XX,
    author = {Gómez, Jairo A. AND Patiño, Jorge E. AND Duque, Juan C. AND Passos, Santiago.},
    journal = {Remote Sensing},
    publisher = {MDPI},
    title = {Spatiotemporal modeling of urban growth using machine learning},
    year = {2020},
    month = {mm},
    volume = {vv},
    url = {xx},
    pages = {xx-xx},
    abstract = {This paper presents a general framework for modeling the growth of three important variables for cities: population distribution, binary urban footprint, and urban footprint in color. The framework models the population distribution as a spatiotemporal regression problem using machine learning, and it obtains the binary urban footprint from the population distribution through a binary classifier plus a temporal correction for existing urban regions. The framework estimates the urban footprint in color from its previous value, as well as from past and current values of the binary urban footprint using a semantic inpainting algorithm. By combining this framework with free data from the Landsat archive and the Global Human Settlement Layer framework, interested users can get approximate growth predictions of any city in the world. These predictions can be improved with the inclusion in the framework of additional spatially distributed input variables over time subject to availability. Unlike widely used growth models based on cellular automata, there are two main advantages of using the proposed machine learning-based framework. Firstly, it does not require to define rules a priori because the model learns the dynamics of growth directly from the historical data. Secondly, it is very easy to train new machine learning models using different explanatory input variables to assess their impact. As a proof of concept, we tested the framework in Valledupar and Rionegro, two Latin American cities located in Colombia with different geomorphological characteristics, and found that the model predictions were in close agreement with the ground-truth based on performance metrics such as the root-mean-square error, zero-mean normalized cross-correlation, Pearson's correlation coefficient for continuous variables, and a few others for discrete variables such as the intersection over union, accuracy, and $f_1$ metric. In summary, our framework for modeling urban growth is flexible, allows sensitivity analyses, and can help policymakers worldwide to assess different what-if scenarios during the planning cycle of sustainable and resilient cities.},
    number = {nn},
    doi = {xx}
}
```
