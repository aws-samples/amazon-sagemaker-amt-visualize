# amt analyze
- Checkout [environment.yml](environment.yml) to create the environment.
- The notebook [RunExperiments.ipynb](RunExperiments.ipynb) is self contained and contains the code to run experiments and to review/analyze the results.


# troubleshooting

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
