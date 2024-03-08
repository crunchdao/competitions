import json
import os
import typing

import click
import jsonschema


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
            file_path = os.path.join(parent, file)
            if not os.path.exists(file_path):
                missing.append(file_path)
        
        if len(missing):
            raise ValueError("some files are missing:\n" + "\n".join(missing))

        print(f"{path}: valid")
        return True
    except Exception as error:
        print(f"{path}: invalid")
        print()
        for line in str(error).splitlines():
            print(f"\t{line}")
        print()

        return False

@click.group()
def cli():
    pass


@cli.command()
@click.option("--quickstarter-json-name", default="quickstarter.json")
@click.option("--schema-path", default="schema/quickstarter.json")
@click.argument("root", default="./competitions/")
def competitions(
    quickstarter_json_name: str,
    schema_path: str,
    root: str
):
    success = True

    with open(schema_path) as fd:
        schema = json.load(fd)

    for competition_name in os.listdir(root):
        competition_root = os.path.join(root, competition_name)
        if not os.path.isdir(competition_root):
            continue

        for quickstarter_name in os.listdir(competition_root):
            quickstarter_root = os.path.join(competition_root, quickstarter_name)
            if not os.path.isdir(quickstarter_root):
                continue

            quickstarter_json_path = os.path.join(quickstarter_root, quickstarter_json_name)
            if not os.path.exists(quickstarter_json_path):
                continue

            if not _validate(schema, quickstarter_json_path):
                success = False

    if not success:
        exit(1)


@cli.command()
@click.option("--quickstarter-json-name", default="quickstarter.json")
@click.option("--schema-path", default="schema/quickstarter.json")
@click.argument("root", default="./generic/")
def generic(
    quickstarter_json_name: str,
    schema_path: str,
    root: str
):
    success = True

    with open(schema_path) as fd:
        schema = json.load(fd)

    for quickstarter_name in os.listdir(root):
        quickstarter_root = os.path.join(root, quickstarter_name)
        if not os.path.isdir(quickstarter_root):
            continue

        quickstarter_json_path = os.path.join(quickstarter_root, quickstarter_json_name)
        if not os.path.exists(quickstarter_json_path):
            continue

        if not _validate(schema, quickstarter_json_path):
            success = False

    if not success:
        exit(1)


if __name__ == '__main__':
    cli()
