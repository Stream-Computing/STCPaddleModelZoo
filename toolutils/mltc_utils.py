# Copyright 2022 Stream Computing Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

def convert_mltc_dtype(name: str):
    dtype_map = {
        "FLOAT64": "f64",
        "FLOAT32": "f32",
        "FLOAT16": "f16",
        "INT64": "i64",
        "INT32": "i32",
        "INT16": "i16",
        "INT8": "i8",
        "UINT64": "ui64",
        "UINT32": "ui32",
        "UINT16": "ui16",
        "UINT8": "ui8"
    }
    if name.upper() in dtype_map.keys():
        return dtype_map[name.upper()]
    
    assert False == True, f"mltc not support dtype convert: {name}"
