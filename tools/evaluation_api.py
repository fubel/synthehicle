import argparse
import requests
import json
import pandas as pd

parser = argparse.ArgumentParser(description="Synthehicle API Evaluator.")

parser.add_argument(
    "-a",
    "--auth-token",
    type=str,
    required=False,
    help="Bearer authentification token.",
)

parser.add_argument(
    "-p",
    "--prediction",
    type=str,
    required=False,
    help="Path to zipped(!) prediction json.",
)

parser.add_argument(
    "-t",
    "--exp-token",
    type=str,
    required=False,
    help="Token to obtain experiment results",
)

parser.add_argument(
    "-s",
    "--server",
    type=str,
    required=True,
    help="Evaluation server URL.",
)

parser.add_argument(
    "-o",
    "--output",
    type=str,
    required=False,
    help="Path to output file.",
)


args = parser.parse_args()


# specify the url of the server
url = args.server

if args.prediction:
    if args.auth_token is not None:
        raise ValueError("Please provide an auth token.")

    # open the zip file and read its contents
    with open(args.prediction, "rb") as zip_file:
        zip_contents = zip_file.read()

    # create a dictionary with the file name and contents to be sent as part of the POST request
    files = {"file": ("predictions.zip", zip_contents)}

    # specify the bearer token to be used for authentication
    bearer_token = args.auth_token

    # create a dictionary with the headers for the POST request
    headers = {"Authorization": f"Bearer {bearer_token}"}

    # send the POST request with the file data and headers
    response = requests.post(url + "/upload", files=files, headers=headers)

    # check the response status code to see if the upload was successful
    if response.status_code == 200:
        # if the upload was successful, get the token from the response data
        response_data = response.json()
        token = response_data["token"]

        # print the token for the user
        print(f"Experiment token: {token}")
    else:
        print("Failed to upload file")
else:
    # get results for experiment token
    response = requests.get(url + "/results/" + args.exp_token)

    if response.status_code == 200:
        data = response.json()
        if data["status"] == "Success":
            # use pandas to print results
            print(pd.DataFrame.from_dict(data["results"]))

            # optionally store output
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(data["results"], f)
        else:
            print(data)
    else:
        print(response.data)
