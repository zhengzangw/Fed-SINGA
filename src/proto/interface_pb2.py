# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: interface.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='interface.proto',
  package='interface',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x0finterface.proto\x12\tinterface\"\x9b\x01\n\x0fWeightsExchange\x12\x1e\n\x07op_type\x18\x01 \x01(\x0e\x32\r.interface.Op\x12\x38\n\x07weights\x18\x02 \x03(\x0b\x32\'.interface.WeightsExchange.WeightsEntry\x1a.\n\x0cWeightsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x0c:\x02\x38\x01*%\n\x02Op\x12\x0b\n\x07\x44\x45\x46\x41ULT\x10\x00\x12\x08\n\x04PUSH\x10\x01\x12\x08\n\x04PULL\x10\x02\x62\x06proto3')
)

_OP = _descriptor.EnumDescriptor(
  name='Op',
  full_name='interface.Op',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='DEFAULT', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PUSH', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PULL', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=188,
  serialized_end=225,
)
_sym_db.RegisterEnumDescriptor(_OP)

Op = enum_type_wrapper.EnumTypeWrapper(_OP)
DEFAULT = 0
PUSH = 1
PULL = 2



_WEIGHTSEXCHANGE_WEIGHTSENTRY = _descriptor.Descriptor(
  name='WeightsEntry',
  full_name='interface.WeightsExchange.WeightsEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='interface.WeightsExchange.WeightsEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='interface.WeightsExchange.WeightsEntry.value', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=140,
  serialized_end=186,
)

_WEIGHTSEXCHANGE = _descriptor.Descriptor(
  name='WeightsExchange',
  full_name='interface.WeightsExchange',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='op_type', full_name='interface.WeightsExchange.op_type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='weights', full_name='interface.WeightsExchange.weights', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_WEIGHTSEXCHANGE_WEIGHTSENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=31,
  serialized_end=186,
)

_WEIGHTSEXCHANGE_WEIGHTSENTRY.containing_type = _WEIGHTSEXCHANGE
_WEIGHTSEXCHANGE.fields_by_name['op_type'].enum_type = _OP
_WEIGHTSEXCHANGE.fields_by_name['weights'].message_type = _WEIGHTSEXCHANGE_WEIGHTSENTRY
DESCRIPTOR.message_types_by_name['WeightsExchange'] = _WEIGHTSEXCHANGE
DESCRIPTOR.enum_types_by_name['Op'] = _OP
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

WeightsExchange = _reflection.GeneratedProtocolMessageType('WeightsExchange', (_message.Message,), {

  'WeightsEntry' : _reflection.GeneratedProtocolMessageType('WeightsEntry', (_message.Message,), {
    'DESCRIPTOR' : _WEIGHTSEXCHANGE_WEIGHTSENTRY,
    '__module__' : 'interface_pb2'
    # @@protoc_insertion_point(class_scope:interface.WeightsExchange.WeightsEntry)
    })
  ,
  'DESCRIPTOR' : _WEIGHTSEXCHANGE,
  '__module__' : 'interface_pb2'
  # @@protoc_insertion_point(class_scope:interface.WeightsExchange)
  })
_sym_db.RegisterMessage(WeightsExchange)
_sym_db.RegisterMessage(WeightsExchange.WeightsEntry)


_WEIGHTSEXCHANGE_WEIGHTSENTRY._options = None
# @@protoc_insertion_point(module_scope)
