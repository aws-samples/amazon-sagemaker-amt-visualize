#  Amazon SageMaker Automatic Model Tuning - Visualizing and Analyzing Hyperparamter Optimization

In these notebooks we demonstrate how to use Amazon SageMaker Automatic Model Tuning (AMT) to perform hyperparameter optimization. We demonstrate how to schedule AMT jobs, and how to visualize and analyze the results. The notebooks demonstrate how to achieve this for several machine learning use cases, such as for tuning a built-in Amazon SageMaker XGBoost algorithm, tuning a BYOC (bring-your-own-container) job and also tuning neural networks.

# Disclaimer

This project is an contains sample code intended to be used for learning purposes only. Do not use in production.

Be aware that the code samples are not covered by the AWS free tier. 


# Troubleshooting

## Could not assume role

solution: edit "Trust relationships" in IAM page
steps:
1. go to isengard AWS console access
2. go to IAM->Roles->Administrator
3. In the "Trust relationships", add a new statement like the following

```json
{
            "Effect": "Allow",
            "Principal": {
                "Service": "sagemaker.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
}
```


## VegaLite object does not show

1. if using JupyterLab, graphic object should show
2. if using jupyter notebook, change the render to default by `alt.renderers.enable('default')`

# Security


See CONTRIBUTING for more information.

# License 

This library is licensed under the MIT-0 License. See the LICENSE file.