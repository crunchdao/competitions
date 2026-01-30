import json
import os
import typing
import urllib.parse
from collections import defaultdict

import click
import jsonschema
import requests

QUICKSTARTER_JSON_NAME = "quickstarter.json"
SCHEMA_PATH = "schema/quickstarter.json"


def print_tab(input: str):
    for line in str(input).splitlines():
        print(f"\t{line}")


def _validate(
    schema: typing.Any,
    path: str,
) -> bool:
    with open(path) as fd:
        quickstarter_json = json.load(fd)

    try:
        jsonschema.validate(instance=quickstarter_json, schema=schema)

        entrypoint = quickstarter_json["entrypoint"]
        if quickstarter_json["notebook"]:
            files = [entrypoint]
        else:
            files = quickstarter_json["files"]

        if entrypoint not in files:
            raise ValueError("entrypoint not in files list")

        parent = os.path.dirname(path)
        missing = []
        for file in files:
            if file.startswith("https://github.com/"):
                # do not check external files
                continue

            file_path = os.path.join(parent, file)
            if not os.path.exists(file_path):
                missing.append(file_path)

        if len(missing):
            raise ValueError("some files are missing:\n" + "\n".join(missing))

        print(f"{path}: valid")
        return quickstarter_json
    except Exception as error:
        print(f"{path}: invalid")
        print()
        print_tab(str(error))
        print()

        return None


def _validate_directories(
    schema: dict,
    root: str,
    items: typing.List[typing.Dict],
):
    root = os.path.join(root, "quickstarters")
    if not os.path.exists(root):
        return True

    success = True
    for quickstarter_root_name in os.listdir(root):
        quickstarter_root = os.path.join(root, quickstarter_root_name)
        if not os.path.isdir(quickstarter_root):
            continue

        quickstarter_json_path = os.path.join(quickstarter_root, QUICKSTARTER_JSON_NAME)
        if not os.path.exists(quickstarter_json_path):
            continue

        quickstarter_json = _validate(schema, quickstarter_json_path)
        if quickstarter_json is None:
            success = False
        else:
            items.append(quickstarter_json)
            quickstarter_json["name"] = quickstarter_root_name

    return success


@click.command()
@click.option("--competition-root", default="./competitions/")
@click.option("--api-base-url", envvar="API_BASE_URL", default="https://api.hub.crunchdao.com")
@click.option("--contact-api", default=False)
@click.option("--api-key", envvar="CRUNCHDAO_API_KEY", default=None)
@click.option("--debug", is_flag=True)
def cli(
    competition_root: str,
    # ---
    api_base_url: str,
    contact_api: str,
    api_key: str,
    # ---
    debug: bool,
):
    success = True
    grouped_by_competition = defaultdict(list)

    with open(SCHEMA_PATH) as fd:
        schema = json.load(fd)

    for competition_name in os.listdir(competition_root):
        competition_name_root = os.path.join(competition_root, competition_name)
        if not os.path.isdir(competition_name_root):
            continue

        items = []
        if not _validate_directories(schema, competition_name_root, items):
            success = False

        for item in items:
            grouped_by_competition[competition_name].append(item)

    if not success:
        exit(1)

    if debug:
        print(json.dumps(items, indent=4))

    for competition_name, quickstarters in grouped_by_competition.items():
        print(f"{competition_name}: found {len(quickstarters)}")

        if contact_api:
            url = urllib.parse.urljoin(api_base_url, f"/v2/competitions/{competition_name}/quickstarters/~")

            response = requests.post(
                url,
                headers={
                    "Authorization": f"API-Key {api_key}"
                },
                json={
                    "quickstarters": quickstarters,
                },
            )

            if response.status_code != 201:
                print(f"{competition_name}: api: {response}: ")
                print_tab(json.dumps(response.json(), indent=4))
                success = False
            else:
                print(f"{competition_name}: delta: {response.json()}")

    if not success:
        exit(1)


if __name__ == '__main__':
    cli()
