# Pure reader so version validation does not require evaluating build dependencies.
versionFile:
let
  data = builtins.fromJSON (builtins.readFile versionFile);
  keys = [ "major" "minor" "patch" "suffix" ];
  number = value: builtins.match "(0|[1-9][0-9]*)" value != null;
  identifiers = "[0-9A-Za-z-]+(\\.[0-9A-Za-z-]+)*";
  # split returns alternating strings and capture groups.
  split = pattern: value: builtins.filter builtins.isString (builtins.split pattern value);
  prerelease = builtins.head (split "\\+" data.suffix);
  validPrerelease = builtins.all
    (id: builtins.match "0[0-9]+" id == null)
    (split "\\." prerelease);
  valid = builtins.isAttrs data
    && builtins.attrNames data == keys
    && builtins.all (key: builtins.isString data.${key}) keys
    && builtins.all (key: number data.${key}) [ "major" "minor" "patch" ]
    && builtins.elem data.major [ "0" "1" "2" ]
    && builtins.match "(${identifiers})?(\\+${identifiers})?" data.suffix != null
    && validPrerelease;
  separator = if builtins.match "\\+.*" data.suffix != null then "" else "-";
in
if !valid then
  throw "invalid version.json: expected four SemVer string fields with MAJOR <= 2"
else
  "${data.major}.${data.minor}.${data.patch}"
  + (if data.suffix == "" then "" else "${separator}${data.suffix}")
