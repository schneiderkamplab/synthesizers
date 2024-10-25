import base64
import importlib.metadata
import platform
from sys import stderr
from xml.etree import ElementTree as ET

MODEL_TAG = "Model"
VERSION_TAG = "Version"
TYPE_TAG = "Type"
PARAMETERS_TAG = "Parameters"

PACKAGES = (
    'py-synthpop',
    'synthcity',
    'syntheval',
)

def unwrap_model_xml(model_type, xml_data):
    root = ET.fromstring(xml_data)
    assert root.tag == MODEL_TAG
    assert model_type == root.find(TYPE_TAG).text
    for version in root.findall(VERSION_TAG):
        package = version.get("id")
        if package == "python":
            if version.get("number") != platform.python_version():
                print(f"Warning: model was trained with python {version.get('number')}, but current Python version is {platform.python_version()}", file=stderr)
        else:
            if version.get("number") != importlib.metadata.version(package):
                print(f"Warning: model was trained with {package} {version.get('number')}, but current {package} version is {importlib.metadata.version(package)}", file=stderr)
    model_data = base64.b64decode(root.find(PARAMETERS_TAG).text)
    return model_data

def wrap_model_xml(model_type, model_data):
    root = ET.Element(MODEL_TAG)
    model_type_elem = ET.SubElement(root, TYPE_TAG)
    model_type_elem.text = model_type
    ET.SubElement(root, VERSION_TAG, id="python", number=platform.python_version())
    for PACKAGE in PACKAGES:
        ET.SubElement(root, VERSION_TAG, id=PACKAGE, number=importlib.metadata.version(PACKAGE))
    model_data_elem = ET.SubElement(root, PARAMETERS_TAG)
    model_data_elem.text = base64.b64encode(model_data).decode("utf-8")
    return ET.tostring(root, encoding="utf-8", method="xml")