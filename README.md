#  Amazon SageMaker Automatic Model Tuning - Visualizing and Analyzing Hyperparameter Optimization

In these notebooks we demonstrate how to use Amazon SageMaker Automatic Model Tuning (AMT) to perform hyperparameter optimization and to interpret the results. We demonstrate how to schedule AMT jobs, and how to visualize and analyze the results. The notebooks demonstrate how to achieve this for several machine learning use cases, such as for [tuning a built-in Amazon SageMaker XGBoost algorithm](1_tuning_of_builtin_xgboost.ipynb), tuning a job with custom training code (future) and also optimizing neural networks (more distant future).

To give you a tiny spoiler on what you will achieve, our solution lets you explore the outcomes of your AMT jobs with interactive graphs depicting the various combinations of hyperparameters:

![AMT graphs 1](/img/amt-gif-1.gif)
![AMT graphs 2](/img/amt-gif-2.gif)

Let's tune!

# Disclaimer

This project contains sample code intended to be used for learning purposes only. Do not use in production.

Be aware that running code samples invoke various AWS services and might involve costs that are not covered by the AWS free tier - please see the [AWS Pricing page](https://aws.amazon.com/pricing/) for details. You are responsible for any AWS costs incurred.

---

# Overview

## Getting Started

To get Amazon SageMaker AMT up and running in your own AWS account, make sure that you have an AWS account (https://aws.amazon.com/getting-started/guides/setup-environment/module-one/) with an Amazon SageMaker Studio (https://docs.aws.amazon.com/sagemaker/latest/dg/studio.html) domain. If you need help creating Amazon SageMaker domain, the procedures of quick or standard setup are described here (https://docs.aws.amazon.com/sagemaker/latest/dg/gs-studio-onboard.html). 

1. Log into the [Amazon SageMaker console](https://console.aws.amazon.com/sagemaker/) if you are not already.
Note: If you are logged in as an IAM user, ensure your account has permissions to create and manage the necessary resources and components for this application.

2. Launch Amazon SageMaker Studio. To do this click on Studio and then on Launch SageMaker Studio:

![SageMaker Studio 1](/img/open_sm_studio_1.png)

Now, click the Launch app button and right-click Studio and Open Link in New Tab (the first time you open Amazon SageMaker Studio this step might take few minutes to launch).

![SageMaker Studio 2](/img/open_sm_studio_2.png)

3. Click the Git icon (step 1 in the picture) and then select Clone a repository (step 2).

![SageMaker Studio Git](/img/smstudio_clone_repo_steps.jpeg)

You'll be asked to type your GitHub username and password, please follow the steps from [here](https://docs.aws.amazon.com/sagemaker/latest/dg/nbi-git-resource.html) to add GitHub repository for the first time you add it. afterwards, you can find your credentials in [AWS Secrets Manager](https://console.aws.amazon.com/secretsmanager).

4. Type in the URL of this repository (`https://github.com/aws-samples/amazon-sagemaker-amt-visualize.git`) and click Clone.

5. Run the notebooks from this repository by opening them from within the File Browser and, once the kernel has started, executing the code cells. Have fun!
A recommended kernel to use with these notebooks is `Python 3` with the image type `Data Science`. 

6. Once you're done, shut down resources running in Amazon SageMaker Studio. Additional guidance for this can be found on this page: https://docs.aws.amazon.com/sagemaker/latest/dg/notebooks-run-and-manage-shut-down.html. 

---

# Troubleshooting

## Could not assume role

To solve this issue, please edit "Trust relationships" on the [IAM](https://console.aws.amazon.com/iam) page. For this:
1. Go to [IAM](https://console.aws.amazon.com/iam)->Roles->Administrator
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

1. The graphic object should display in both JupyterLab (versions 1.0 and later) on Amazon SageMaker Studio as well as when using a Jupyter Notebook with SageMaker Notebooks
2. If you see `<VegaLite 3 object>` when running in a JupyterLab environment, please ensure that your JupyterLab version is 1.0 or later
3. If you see `<VegaLite 3 object>` when running in a Jupyter Notebook, change the renderer to `notebook` by uncommenting `alt.renderers.enable('notebook')` in `amtviz/visualization.py`. If the problem persists, ensure that you are running a Jupyter Notebook at version 5.3 or later and install version 2.0 or later of the vega package, by running  `pip install "vega>=2.0"`
4. To enable charts to be displayed statically, on pages such as GitHub, you can change the renderer to `mimetype` by uncommenting `alt.renderers.enable('mimetype')` in `amtviz/visualization.py`. Please note that you must be running a JupyterLab 3 (or later) Notebook locally or with SageMaker Notebooks in this case.

---

# Security

Although we're extremely excited to receive contributions from the community, we're still working on the best mechanism to take in examples from external sources. Please bear with us in the short-term if pull requests take longer than expected or are closed. Please read our [CONTRIBUTING](https://github.com/aws-samples/amazon-sagemaker-amt-visualize/blob/main/CONTRIBUTING.md) guidelines if you'd like to open an issue or submit a pull request.

---

# License 

This library is licensed under the MIT-0 License. See the [LICENSE](https://github.com/aws-samples/amazon-sagemaker-amt-visualize/blob/main/LICENSE) file.