"""
Configuration settings for ORKG Template Creator
"""

# ORKG Connection Settings
ORKG_HOST = "https://orkg.org"
ORKG_USERNAME = "" # Your ORKG username
ORKG_PASSWORD = "" # Your ORKG password

# ORKG Predicate IDs (commonly used)
PREDICATES = {
    "sh:targetClass": "sh:targetClass",
    "sh:property": "sh:property",
    "sh:path": "sh:path",
    "sh:datatype": "sh:datatype",
    "sh:class": "sh:class",
    "sh:maxCount": "sh:maxCount",
}

# ORKG Class IDs (commonly used)
CLASSES = {"NodeShape": "NodeShape", "PropertyShape": "PropertyShape"}

# Data types
DATATYPES = {"String": "xsd:string", "xsd:integer": "xsd:integer"}
