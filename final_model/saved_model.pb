??#
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-0-ga4dfb8d1a718??
?
watermark_hidden1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??@*)
shared_namewatermark_hidden1/kernel
?
,watermark_hidden1/kernel/Read/ReadVariableOpReadVariableOpwatermark_hidden1/kernel* 
_output_shapes
:
??@*
dtype0
?
watermark_hidden1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_namewatermark_hidden1/bias
}
*watermark_hidden1/bias/Read/ReadVariableOpReadVariableOpwatermark_hidden1/bias*
_output_shapes
:@*
dtype0
?
cartoon_hidden1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??@*'
shared_namecartoon_hidden1/kernel
?
*cartoon_hidden1/kernel/Read/ReadVariableOpReadVariableOpcartoon_hidden1/kernel* 
_output_shapes
:
??@*
dtype0
?
cartoon_hidden1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_namecartoon_hidden1/bias
y
(cartoon_hidden1/bias/Read/ReadVariableOpReadVariableOpcartoon_hidden1/bias*
_output_shapes
:@*
dtype0
?
watermark_hidden2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *)
shared_namewatermark_hidden2/kernel
?
,watermark_hidden2/kernel/Read/ReadVariableOpReadVariableOpwatermark_hidden2/kernel*
_output_shapes

:@ *
dtype0
?
watermark_hidden2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namewatermark_hidden2/bias
}
*watermark_hidden2/bias/Read/ReadVariableOpReadVariableOpwatermark_hidden2/bias*
_output_shapes
: *
dtype0
?
cartoon_hidden2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_namecartoon_hidden2/kernel
?
*cartoon_hidden2/kernel/Read/ReadVariableOpReadVariableOpcartoon_hidden2/kernel*
_output_shapes

:@*
dtype0
?
cartoon_hidden2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namecartoon_hidden2/bias
y
(cartoon_hidden2/bias/Read/ReadVariableOpReadVariableOpcartoon_hidden2/bias*
_output_shapes
:*
dtype0
?
watermark_hidden3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *)
shared_namewatermark_hidden3/kernel
?
,watermark_hidden3/kernel/Read/ReadVariableOpReadVariableOpwatermark_hidden3/kernel*
_output_shapes

: *
dtype0
?
watermark_hidden3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namewatermark_hidden3/bias
}
*watermark_hidden3/bias/Read/ReadVariableOpReadVariableOpwatermark_hidden3/bias*
_output_shapes
:*
dtype0
?
cartoon_hidden3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_namecartoon_hidden3/kernel
?
*cartoon_hidden3/kernel/Read/ReadVariableOpReadVariableOpcartoon_hidden3/kernel*
_output_shapes

:
*
dtype0
?
cartoon_hidden3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_namecartoon_hidden3/bias
y
(cartoon_hidden3/bias/Read/ReadVariableOpReadVariableOpcartoon_hidden3/bias*
_output_shapes
:
*
dtype0
?
watermark_hidden4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*)
shared_namewatermark_hidden4/kernel
?
,watermark_hidden4/kernel/Read/ReadVariableOpReadVariableOpwatermark_hidden4/kernel*
_output_shapes

:
*
dtype0
?
watermark_hidden4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_namewatermark_hidden4/bias
}
*watermark_hidden4/bias/Read/ReadVariableOpReadVariableOpwatermark_hidden4/bias*
_output_shapes
:
*
dtype0
?
cartoon_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*&
shared_namecartoon_output/kernel

)cartoon_output/kernel/Read/ReadVariableOpReadVariableOpcartoon_output/kernel*
_output_shapes

:
*
dtype0
~
cartoon_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namecartoon_output/bias
w
'cartoon_output/bias/Read/ReadVariableOpReadVariableOpcartoon_output/bias*
_output_shapes
:*
dtype0
?
watermark_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*(
shared_namewatermark_output/kernel
?
+watermark_output/kernel/Read/ReadVariableOpReadVariableOpwatermark_output/kernel*
_output_shapes

:
*
dtype0
?
watermark_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namewatermark_output/bias
{
)watermark_output/bias/Read/ReadVariableOpReadVariableOpwatermark_output/bias*
_output_shapes
:*
dtype0
?
block1_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameblock1_conv1/kernel
?
'block1_conv1/kernel/Read/ReadVariableOpReadVariableOpblock1_conv1/kernel*&
_output_shapes
:@*
dtype0
z
block1_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv1/bias
s
%block1_conv1/bias/Read/ReadVariableOpReadVariableOpblock1_conv1/bias*
_output_shapes
:@*
dtype0
?
block1_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*$
shared_nameblock1_conv2/kernel
?
'block1_conv2/kernel/Read/ReadVariableOpReadVariableOpblock1_conv2/kernel*&
_output_shapes
:@@*
dtype0
z
block1_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv2/bias
s
%block1_conv2/bias/Read/ReadVariableOpReadVariableOpblock1_conv2/bias*
_output_shapes
:@*
dtype0
?
block2_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*$
shared_nameblock2_conv1/kernel
?
'block2_conv1/kernel/Read/ReadVariableOpReadVariableOpblock2_conv1/kernel*'
_output_shapes
:@?*
dtype0
{
block2_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock2_conv1/bias
t
%block2_conv1/bias/Read/ReadVariableOpReadVariableOpblock2_conv1/bias*
_output_shapes	
:?*
dtype0
?
block2_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock2_conv2/kernel
?
'block2_conv2/kernel/Read/ReadVariableOpReadVariableOpblock2_conv2/kernel*(
_output_shapes
:??*
dtype0
{
block2_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock2_conv2/bias
t
%block2_conv2/bias/Read/ReadVariableOpReadVariableOpblock2_conv2/bias*
_output_shapes	
:?*
dtype0
?
block3_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock3_conv1/kernel
?
'block3_conv1/kernel/Read/ReadVariableOpReadVariableOpblock3_conv1/kernel*(
_output_shapes
:??*
dtype0
{
block3_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock3_conv1/bias
t
%block3_conv1/bias/Read/ReadVariableOpReadVariableOpblock3_conv1/bias*
_output_shapes	
:?*
dtype0
?
block3_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock3_conv2/kernel
?
'block3_conv2/kernel/Read/ReadVariableOpReadVariableOpblock3_conv2/kernel*(
_output_shapes
:??*
dtype0
{
block3_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock3_conv2/bias
t
%block3_conv2/bias/Read/ReadVariableOpReadVariableOpblock3_conv2/bias*
_output_shapes	
:?*
dtype0
?
block3_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock3_conv3/kernel
?
'block3_conv3/kernel/Read/ReadVariableOpReadVariableOpblock3_conv3/kernel*(
_output_shapes
:??*
dtype0
{
block3_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock3_conv3/bias
t
%block3_conv3/bias/Read/ReadVariableOpReadVariableOpblock3_conv3/bias*
_output_shapes	
:?*
dtype0
?
block4_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock4_conv1/kernel
?
'block4_conv1/kernel/Read/ReadVariableOpReadVariableOpblock4_conv1/kernel*(
_output_shapes
:??*
dtype0
{
block4_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock4_conv1/bias
t
%block4_conv1/bias/Read/ReadVariableOpReadVariableOpblock4_conv1/bias*
_output_shapes	
:?*
dtype0
?
block4_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock4_conv2/kernel
?
'block4_conv2/kernel/Read/ReadVariableOpReadVariableOpblock4_conv2/kernel*(
_output_shapes
:??*
dtype0
{
block4_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock4_conv2/bias
t
%block4_conv2/bias/Read/ReadVariableOpReadVariableOpblock4_conv2/bias*
_output_shapes	
:?*
dtype0
?
block4_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock4_conv3/kernel
?
'block4_conv3/kernel/Read/ReadVariableOpReadVariableOpblock4_conv3/kernel*(
_output_shapes
:??*
dtype0
{
block4_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock4_conv3/bias
t
%block4_conv3/bias/Read/ReadVariableOpReadVariableOpblock4_conv3/bias*
_output_shapes	
:?*
dtype0
?
block5_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock5_conv1/kernel
?
'block5_conv1/kernel/Read/ReadVariableOpReadVariableOpblock5_conv1/kernel*(
_output_shapes
:??*
dtype0
{
block5_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock5_conv1/bias
t
%block5_conv1/bias/Read/ReadVariableOpReadVariableOpblock5_conv1/bias*
_output_shapes	
:?*
dtype0
?
block5_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock5_conv2/kernel
?
'block5_conv2/kernel/Read/ReadVariableOpReadVariableOpblock5_conv2/kernel*(
_output_shapes
:??*
dtype0
{
block5_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock5_conv2/bias
t
%block5_conv2/bias/Read/ReadVariableOpReadVariableOpblock5_conv2/bias*
_output_shapes	
:?*
dtype0
?
block5_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock5_conv3/kernel
?
'block5_conv3/kernel/Read/ReadVariableOpReadVariableOpblock5_conv3/kernel*(
_output_shapes
:??*
dtype0
{
block5_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock5_conv3/bias
t
%block5_conv3/bias/Read/ReadVariableOpReadVariableOpblock5_conv3/bias*
_output_shapes	
:?*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer_with_weights-9
layer-12
	optimizer
loss
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 

	keras_api
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
layer_with_weights-5
layer-8
layer_with_weights-6
layer-9
 layer-10
!layer_with_weights-7
!layer-11
"layer_with_weights-8
"layer-12
#layer_with_weights-9
#layer-13
$layer-14
%layer_with_weights-10
%layer-15
&layer_with_weights-11
&layer-16
'layer_with_weights-12
'layer-17
(layer-18
)regularization_losses
*trainable_variables
+	variables
,	keras_api
R
-regularization_losses
.trainable_variables
/	variables
0	keras_api
h

1kernel
2bias
3regularization_losses
4trainable_variables
5	variables
6	keras_api
h

7kernel
8bias
9regularization_losses
:trainable_variables
;	variables
<	keras_api
h

=kernel
>bias
?regularization_losses
@trainable_variables
A	variables
B	keras_api
h

Ckernel
Dbias
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
h

Ikernel
Jbias
Kregularization_losses
Ltrainable_variables
M	variables
N	keras_api
h

Okernel
Pbias
Qregularization_losses
Rtrainable_variables
S	variables
T	keras_api
h

Ukernel
Vbias
Wregularization_losses
Xtrainable_variables
Y	variables
Z	keras_api
h

[kernel
\bias
]regularization_losses
^trainable_variables
_	variables
`	keras_api
h

akernel
bbias
cregularization_losses
dtrainable_variables
e	variables
f	keras_api
 
 
 
?
10
21
72
83
=4
>5
C6
D7
I8
J9
O10
P11
U12
V13
[14
\15
a16
b17
?
g0
h1
i2
j3
k4
l5
m6
n7
o8
p9
q10
r11
s12
t13
u14
v15
w16
x17
y18
z19
{20
|21
}22
~23
24
?25
126
227
728
829
=30
>31
C32
D33
I34
J35
O36
P37
U38
V39
[40
\41
a42
b43
?
regularization_losses
?layer_metrics
trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
	variables
 
 
 
l

gkernel
hbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
l

ikernel
jbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
l

kkernel
lbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
l

mkernel
nbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
l

okernel
pbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
l

qkernel
rbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
l

skernel
tbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
l

ukernel
vbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
l

wkernel
xbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
l

ykernel
zbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
l

{kernel
|bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
l

}kernel
~bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
m

kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
 
 
?
g0
h1
i2
j3
k4
l5
m6
n7
o8
p9
q10
r11
s12
t13
u14
v15
w16
x17
y18
z19
{20
|21
}22
~23
24
?25
?
)regularization_losses
?layer_metrics
*trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
+	variables
 
 
 
?
-regularization_losses
?layer_metrics
.trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
/	variables
db
VARIABLE_VALUEwatermark_hidden1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEwatermark_hidden1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

10
21

10
21
?
3regularization_losses
?layer_metrics
4trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
5	variables
b`
VARIABLE_VALUEcartoon_hidden1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEcartoon_hidden1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

70
81

70
81
?
9regularization_losses
?layer_metrics
:trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
;	variables
db
VARIABLE_VALUEwatermark_hidden2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEwatermark_hidden2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

=0
>1

=0
>1
?
?regularization_losses
?layer_metrics
@trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
A	variables
b`
VARIABLE_VALUEcartoon_hidden2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEcartoon_hidden2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

C0
D1

C0
D1
?
Eregularization_losses
?layer_metrics
Ftrainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
G	variables
db
VARIABLE_VALUEwatermark_hidden3/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEwatermark_hidden3/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

I0
J1

I0
J1
?
Kregularization_losses
?layer_metrics
Ltrainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
M	variables
b`
VARIABLE_VALUEcartoon_hidden3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEcartoon_hidden3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

O0
P1

O0
P1
?
Qregularization_losses
?layer_metrics
Rtrainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
S	variables
db
VARIABLE_VALUEwatermark_hidden4/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEwatermark_hidden4/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

U0
V1

U0
V1
?
Wregularization_losses
?layer_metrics
Xtrainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
Y	variables
a_
VARIABLE_VALUEcartoon_output/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEcartoon_output/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

[0
\1

[0
\1
?
]regularization_losses
?layer_metrics
^trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
_	variables
ca
VARIABLE_VALUEwatermark_output/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEwatermark_output/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

a0
b1

a0
b1
?
cregularization_losses
?layer_metrics
dtrainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
e	variables
OM
VARIABLE_VALUEblock1_conv1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock1_conv1/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock1_conv2/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock1_conv2/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock2_conv1/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock2_conv1/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock2_conv2/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock2_conv2/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock3_conv1/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock3_conv1/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock3_conv2/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock3_conv2/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock3_conv3/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock3_conv3/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock4_conv1/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock4_conv1/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock4_conv2/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock4_conv2/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock4_conv3/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock4_conv3/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock5_conv1/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock5_conv1/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock5_conv2/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock5_conv2/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock5_conv3/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock5_conv3/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
 
 
?
g0
h1
i2
j3
k4
l5
m6
n7
o8
p9
q10
r11
s12
t13
u14
v15
w16
x17
y18
z19
{20
|21
}22
~23
24
?25

?0
?1
^
0
1
2
3
4
5
6
7
	8

9
10
11
12
 
 

g0
h1
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
?	variables
 
 

i0
j1
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
?	variables
 
 
 
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
?	variables
 
 

k0
l1
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
?	variables
 
 

m0
n1
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
?	variables
 
 
 
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
?	variables
 
 

o0
p1
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
?	variables
 
 

q0
r1
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
?	variables
 
 

s0
t1
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
?	variables
 
 
 
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
?	variables
 
 

u0
v1
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
?	variables
 
 

w0
x1
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
?	variables
 
 

y0
z1
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
?	variables
 
 
 
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
?	variables
 
 

{0
|1
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
?	variables
 
 

}0
~1
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
?	variables
 
 

0
?1
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
?	variables
 
 
 
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
?	variables
 
 
?
g0
h1
i2
j3
k4
l5
m6
n7
o8
p9
q10
r11
s12
t13
u14
v15
w16
x17
y18
z19
{20
|21
}22
~23
24
?25
 
?
0
1
2
3
4
5
6
7
8
9
 10
!11
"12
#13
$14
%15
&16
'17
(18
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
 
 

g0
h1
 
 
 
 

i0
j1
 
 
 
 
 
 
 
 
 

k0
l1
 
 
 
 

m0
n1
 
 
 
 
 
 
 
 
 

o0
p1
 
 
 
 

q0
r1
 
 
 
 

s0
t1
 
 
 
 
 
 
 
 
 

u0
v1
 
 
 
 

w0
x1
 
 
 
 

y0
z1
 
 
 
 
 
 
 
 
 

{0
|1
 
 
 
 

}0
~1
 
 
 
 

0
?1
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
?
serving_default_input_31Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_31block1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/biasblock5_conv3/kernelblock5_conv3/biaswatermark_hidden1/kernelwatermark_hidden1/biaswatermark_hidden2/kernelwatermark_hidden2/biascartoon_hidden1/kernelcartoon_hidden1/biaswatermark_hidden3/kernelwatermark_hidden3/biascartoon_hidden2/kernelcartoon_hidden2/biaswatermark_hidden4/kernelwatermark_hidden4/biascartoon_hidden3/kernelcartoon_hidden3/biaswatermark_output/kernelwatermark_output/biascartoon_output/kernelcartoon_output/bias*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_108041
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename,watermark_hidden1/kernel/Read/ReadVariableOp*watermark_hidden1/bias/Read/ReadVariableOp*cartoon_hidden1/kernel/Read/ReadVariableOp(cartoon_hidden1/bias/Read/ReadVariableOp,watermark_hidden2/kernel/Read/ReadVariableOp*watermark_hidden2/bias/Read/ReadVariableOp*cartoon_hidden2/kernel/Read/ReadVariableOp(cartoon_hidden2/bias/Read/ReadVariableOp,watermark_hidden3/kernel/Read/ReadVariableOp*watermark_hidden3/bias/Read/ReadVariableOp*cartoon_hidden3/kernel/Read/ReadVariableOp(cartoon_hidden3/bias/Read/ReadVariableOp,watermark_hidden4/kernel/Read/ReadVariableOp*watermark_hidden4/bias/Read/ReadVariableOp)cartoon_output/kernel/Read/ReadVariableOp'cartoon_output/bias/Read/ReadVariableOp+watermark_output/kernel/Read/ReadVariableOp)watermark_output/bias/Read/ReadVariableOp'block1_conv1/kernel/Read/ReadVariableOp%block1_conv1/bias/Read/ReadVariableOp'block1_conv2/kernel/Read/ReadVariableOp%block1_conv2/bias/Read/ReadVariableOp'block2_conv1/kernel/Read/ReadVariableOp%block2_conv1/bias/Read/ReadVariableOp'block2_conv2/kernel/Read/ReadVariableOp%block2_conv2/bias/Read/ReadVariableOp'block3_conv1/kernel/Read/ReadVariableOp%block3_conv1/bias/Read/ReadVariableOp'block3_conv2/kernel/Read/ReadVariableOp%block3_conv2/bias/Read/ReadVariableOp'block3_conv3/kernel/Read/ReadVariableOp%block3_conv3/bias/Read/ReadVariableOp'block4_conv1/kernel/Read/ReadVariableOp%block4_conv1/bias/Read/ReadVariableOp'block4_conv2/kernel/Read/ReadVariableOp%block4_conv2/bias/Read/ReadVariableOp'block4_conv3/kernel/Read/ReadVariableOp%block4_conv3/bias/Read/ReadVariableOp'block5_conv1/kernel/Read/ReadVariableOp%block5_conv1/bias/Read/ReadVariableOp'block5_conv2/kernel/Read/ReadVariableOp%block5_conv2/bias/Read/ReadVariableOp'block5_conv3/kernel/Read/ReadVariableOp%block5_conv3/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst*=
Tin6
422*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_save_109504
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamewatermark_hidden1/kernelwatermark_hidden1/biascartoon_hidden1/kernelcartoon_hidden1/biaswatermark_hidden2/kernelwatermark_hidden2/biascartoon_hidden2/kernelcartoon_hidden2/biaswatermark_hidden3/kernelwatermark_hidden3/biascartoon_hidden3/kernelcartoon_hidden3/biaswatermark_hidden4/kernelwatermark_hidden4/biascartoon_output/kernelcartoon_output/biaswatermark_output/kernelwatermark_output/biasblock1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/biasblock5_conv3/kernelblock5_conv3/biastotalcounttotal_1count_1*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_restore_109658??
?
?
H__inference_block2_conv2_layer_call_and_return_conditional_losses_106177

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2	
BiasAddc
ReluReluBiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
c
G__inference_block3_pool_layer_call_and_return_conditional_losses_106077

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
b
F__inference_flatten_28_layer_call_and_return_conditional_losses_106989

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ?  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????		?:X T
0
_output_shapes
:?????????		?
 
_user_specified_nameinputs
?R
?
D__inference_model_28_layer_call_and_return_conditional_losses_107944
input_31&
vgg16_107843:@
vgg16_107845:@&
vgg16_107847:@@
vgg16_107849:@'
vgg16_107851:@?
vgg16_107853:	?(
vgg16_107855:??
vgg16_107857:	?(
vgg16_107859:??
vgg16_107861:	?(
vgg16_107863:??
vgg16_107865:	?(
vgg16_107867:??
vgg16_107869:	?(
vgg16_107871:??
vgg16_107873:	?(
vgg16_107875:??
vgg16_107877:	?(
vgg16_107879:??
vgg16_107881:	?(
vgg16_107883:??
vgg16_107885:	?(
vgg16_107887:??
vgg16_107889:	?(
vgg16_107891:??
vgg16_107893:	?,
watermark_hidden1_107897:
??@&
watermark_hidden1_107899:@*
watermark_hidden2_107902:@ &
watermark_hidden2_107904: *
cartoon_hidden1_107907:
??@$
cartoon_hidden1_107909:@*
watermark_hidden3_107912: &
watermark_hidden3_107914:(
cartoon_hidden2_107917:@$
cartoon_hidden2_107919:*
watermark_hidden4_107922:
&
watermark_hidden4_107924:
(
cartoon_hidden3_107927:
$
cartoon_hidden3_107929:
)
watermark_output_107932:
%
watermark_output_107934:'
cartoon_output_107937:
#
cartoon_output_107939:
identity

identity_1??'cartoon_hidden1/StatefulPartitionedCall?'cartoon_hidden2/StatefulPartitionedCall?'cartoon_hidden3/StatefulPartitionedCall?&cartoon_output/StatefulPartitionedCall?vgg16/StatefulPartitionedCall?)watermark_hidden1/StatefulPartitionedCall?)watermark_hidden2/StatefulPartitionedCall?)watermark_hidden3/StatefulPartitionedCall?)watermark_hidden4/StatefulPartitionedCall?(watermark_output/StatefulPartitionedCallo
rescaling_24/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???;2
rescaling_24/Cast/xs
rescaling_24/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling_24/Cast_1/x?
rescaling_24/mulMulinput_31rescaling_24/Cast/x:output:0*
T0*1
_output_shapes
:???????????2
rescaling_24/mul?
rescaling_24/addAddV2rescaling_24/mul:z:0rescaling_24/Cast_1/x:output:0*
T0*1
_output_shapes
:???????????2
rescaling_24/add?
vgg16/StatefulPartitionedCallStatefulPartitionedCallrescaling_24/add:z:0vgg16_107843vgg16_107845vgg16_107847vgg16_107849vgg16_107851vgg16_107853vgg16_107855vgg16_107857vgg16_107859vgg16_107861vgg16_107863vgg16_107865vgg16_107867vgg16_107869vgg16_107871vgg16_107873vgg16_107875vgg16_107877vgg16_107879vgg16_107881vgg16_107883vgg16_107885vgg16_107887vgg16_107889vgg16_107891vgg16_107893*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_vgg16_layer_call_and_return_conditional_losses_1066592
vgg16/StatefulPartitionedCall?
flatten_28/PartitionedCallPartitionedCall&vgg16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_28_layer_call_and_return_conditional_losses_1069892
flatten_28/PartitionedCall?
)watermark_hidden1/StatefulPartitionedCallStatefulPartitionedCall#flatten_28/PartitionedCall:output:0watermark_hidden1_107897watermark_hidden1_107899*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_watermark_hidden1_layer_call_and_return_conditional_losses_1070022+
)watermark_hidden1/StatefulPartitionedCall?
)watermark_hidden2/StatefulPartitionedCallStatefulPartitionedCall2watermark_hidden1/StatefulPartitionedCall:output:0watermark_hidden2_107902watermark_hidden2_107904*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_watermark_hidden2_layer_call_and_return_conditional_losses_1070192+
)watermark_hidden2/StatefulPartitionedCall?
'cartoon_hidden1/StatefulPartitionedCallStatefulPartitionedCall#flatten_28/PartitionedCall:output:0cartoon_hidden1_107907cartoon_hidden1_107909*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_cartoon_hidden1_layer_call_and_return_conditional_losses_1070362)
'cartoon_hidden1/StatefulPartitionedCall?
)watermark_hidden3/StatefulPartitionedCallStatefulPartitionedCall2watermark_hidden2/StatefulPartitionedCall:output:0watermark_hidden3_107912watermark_hidden3_107914*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_watermark_hidden3_layer_call_and_return_conditional_losses_1070532+
)watermark_hidden3/StatefulPartitionedCall?
'cartoon_hidden2/StatefulPartitionedCallStatefulPartitionedCall0cartoon_hidden1/StatefulPartitionedCall:output:0cartoon_hidden2_107917cartoon_hidden2_107919*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_cartoon_hidden2_layer_call_and_return_conditional_losses_1070702)
'cartoon_hidden2/StatefulPartitionedCall?
)watermark_hidden4/StatefulPartitionedCallStatefulPartitionedCall2watermark_hidden3/StatefulPartitionedCall:output:0watermark_hidden4_107922watermark_hidden4_107924*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_watermark_hidden4_layer_call_and_return_conditional_losses_1070872+
)watermark_hidden4/StatefulPartitionedCall?
'cartoon_hidden3/StatefulPartitionedCallStatefulPartitionedCall0cartoon_hidden2/StatefulPartitionedCall:output:0cartoon_hidden3_107927cartoon_hidden3_107929*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_cartoon_hidden3_layer_call_and_return_conditional_losses_1071042)
'cartoon_hidden3/StatefulPartitionedCall?
(watermark_output/StatefulPartitionedCallStatefulPartitionedCall2watermark_hidden4/StatefulPartitionedCall:output:0watermark_output_107932watermark_output_107934*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_watermark_output_layer_call_and_return_conditional_losses_1071212*
(watermark_output/StatefulPartitionedCall?
&cartoon_output/StatefulPartitionedCallStatefulPartitionedCall0cartoon_hidden3/StatefulPartitionedCall:output:0cartoon_output_107937cartoon_output_107939*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_cartoon_output_layer_call_and_return_conditional_losses_1071382(
&cartoon_output/StatefulPartitionedCall?
IdentityIdentity/cartoon_output/StatefulPartitionedCall:output:0(^cartoon_hidden1/StatefulPartitionedCall(^cartoon_hidden2/StatefulPartitionedCall(^cartoon_hidden3/StatefulPartitionedCall'^cartoon_output/StatefulPartitionedCall^vgg16/StatefulPartitionedCall*^watermark_hidden1/StatefulPartitionedCall*^watermark_hidden2/StatefulPartitionedCall*^watermark_hidden3/StatefulPartitionedCall*^watermark_hidden4/StatefulPartitionedCall)^watermark_output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity1watermark_output/StatefulPartitionedCall:output:0(^cartoon_hidden1/StatefulPartitionedCall(^cartoon_hidden2/StatefulPartitionedCall(^cartoon_hidden3/StatefulPartitionedCall'^cartoon_output/StatefulPartitionedCall^vgg16/StatefulPartitionedCall*^watermark_hidden1/StatefulPartitionedCall*^watermark_hidden2/StatefulPartitionedCall*^watermark_hidden3/StatefulPartitionedCall*^watermark_hidden4/StatefulPartitionedCall)^watermark_output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesw
u:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2R
'cartoon_hidden1/StatefulPartitionedCall'cartoon_hidden1/StatefulPartitionedCall2R
'cartoon_hidden2/StatefulPartitionedCall'cartoon_hidden2/StatefulPartitionedCall2R
'cartoon_hidden3/StatefulPartitionedCall'cartoon_hidden3/StatefulPartitionedCall2P
&cartoon_output/StatefulPartitionedCall&cartoon_output/StatefulPartitionedCall2>
vgg16/StatefulPartitionedCallvgg16/StatefulPartitionedCall2V
)watermark_hidden1/StatefulPartitionedCall)watermark_hidden1/StatefulPartitionedCall2V
)watermark_hidden2/StatefulPartitionedCall)watermark_hidden2/StatefulPartitionedCall2V
)watermark_hidden3/StatefulPartitionedCall)watermark_hidden3/StatefulPartitionedCall2V
)watermark_hidden4/StatefulPartitionedCall)watermark_hidden4/StatefulPartitionedCall2T
(watermark_output/StatefulPartitionedCall(watermark_output/StatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_31
?\
?
A__inference_vgg16_layer_call_and_return_conditional_losses_106659

inputs-
block1_conv1_106588:@!
block1_conv1_106590:@-
block1_conv2_106593:@@!
block1_conv2_106595:@.
block2_conv1_106599:@?"
block2_conv1_106601:	?/
block2_conv2_106604:??"
block2_conv2_106606:	?/
block3_conv1_106610:??"
block3_conv1_106612:	?/
block3_conv2_106615:??"
block3_conv2_106617:	?/
block3_conv3_106620:??"
block3_conv3_106622:	?/
block4_conv1_106626:??"
block4_conv1_106628:	?/
block4_conv2_106631:??"
block4_conv2_106633:	?/
block4_conv3_106636:??"
block4_conv3_106638:	?/
block5_conv1_106642:??"
block5_conv1_106644:	?/
block5_conv2_106647:??"
block5_conv2_106649:	?/
block5_conv3_106652:??"
block5_conv3_106654:	?
identity??$block1_conv1/StatefulPartitionedCall?$block1_conv2/StatefulPartitionedCall?$block2_conv1/StatefulPartitionedCall?$block2_conv2/StatefulPartitionedCall?$block3_conv1/StatefulPartitionedCall?$block3_conv2/StatefulPartitionedCall?$block3_conv3/StatefulPartitionedCall?$block4_conv1/StatefulPartitionedCall?$block4_conv2/StatefulPartitionedCall?$block4_conv3/StatefulPartitionedCall?$block5_conv1/StatefulPartitionedCall?$block5_conv2/StatefulPartitionedCall?$block5_conv3/StatefulPartitionedCall?
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_106588block1_conv1_106590*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block1_conv1_layer_call_and_return_conditional_losses_1061252&
$block1_conv1/StatefulPartitionedCall?
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_106593block1_conv2_106595*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block1_conv2_layer_call_and_return_conditional_losses_1061422&
$block1_conv2/StatefulPartitionedCall?
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block1_pool_layer_call_and_return_conditional_losses_1060532
block1_pool/PartitionedCall?
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_106599block2_conv1_106601*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block2_conv1_layer_call_and_return_conditional_losses_1061602&
$block2_conv1/StatefulPartitionedCall?
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_106604block2_conv2_106606*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block2_conv2_layer_call_and_return_conditional_losses_1061772&
$block2_conv2/StatefulPartitionedCall?
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????KK?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block2_pool_layer_call_and_return_conditional_losses_1060652
block2_pool/PartitionedCall?
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_106610block3_conv1_106612*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????KK?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block3_conv1_layer_call_and_return_conditional_losses_1061952&
$block3_conv1/StatefulPartitionedCall?
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_106615block3_conv2_106617*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????KK?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block3_conv2_layer_call_and_return_conditional_losses_1062122&
$block3_conv2/StatefulPartitionedCall?
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_106620block3_conv3_106622*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????KK?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block3_conv3_layer_call_and_return_conditional_losses_1062292&
$block3_conv3/StatefulPartitionedCall?
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????%%?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_pool_layer_call_and_return_conditional_losses_1060772
block3_pool/PartitionedCall?
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_106626block4_conv1_106628*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????%%?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block4_conv1_layer_call_and_return_conditional_losses_1062472&
$block4_conv1/StatefulPartitionedCall?
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_106631block4_conv2_106633*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????%%?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block4_conv2_layer_call_and_return_conditional_losses_1062642&
$block4_conv2/StatefulPartitionedCall?
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_106636block4_conv3_106638*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????%%?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block4_conv3_layer_call_and_return_conditional_losses_1062812&
$block4_conv3/StatefulPartitionedCall?
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_pool_layer_call_and_return_conditional_losses_1060892
block4_pool/PartitionedCall?
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_106642block5_conv1_106644*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block5_conv1_layer_call_and_return_conditional_losses_1062992&
$block5_conv1/StatefulPartitionedCall?
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_106647block5_conv2_106649*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block5_conv2_layer_call_and_return_conditional_losses_1063162&
$block5_conv2/StatefulPartitionedCall?
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_106652block5_conv3_106654*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block5_conv3_layer_call_and_return_conditional_losses_1063332&
$block5_conv3/StatefulPartitionedCall?
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_pool_layer_call_and_return_conditional_losses_1061012
block5_pool/PartitionedCall?
IdentityIdentity$block5_pool/PartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
L__inference_watermark_output_layer_call_and_return_conditional_losses_109076

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
H__inference_block4_conv1_layer_call_and_return_conditional_losses_106247

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????%%?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????%%?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????%%?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????%%?
 
_user_specified_nameinputs
?\
?
A__inference_vgg16_layer_call_and_return_conditional_losses_106919
input_1-
block1_conv1_106848:@!
block1_conv1_106850:@-
block1_conv2_106853:@@!
block1_conv2_106855:@.
block2_conv1_106859:@?"
block2_conv1_106861:	?/
block2_conv2_106864:??"
block2_conv2_106866:	?/
block3_conv1_106870:??"
block3_conv1_106872:	?/
block3_conv2_106875:??"
block3_conv2_106877:	?/
block3_conv3_106880:??"
block3_conv3_106882:	?/
block4_conv1_106886:??"
block4_conv1_106888:	?/
block4_conv2_106891:??"
block4_conv2_106893:	?/
block4_conv3_106896:??"
block4_conv3_106898:	?/
block5_conv1_106902:??"
block5_conv1_106904:	?/
block5_conv2_106907:??"
block5_conv2_106909:	?/
block5_conv3_106912:??"
block5_conv3_106914:	?
identity??$block1_conv1/StatefulPartitionedCall?$block1_conv2/StatefulPartitionedCall?$block2_conv1/StatefulPartitionedCall?$block2_conv2/StatefulPartitionedCall?$block3_conv1/StatefulPartitionedCall?$block3_conv2/StatefulPartitionedCall?$block3_conv3/StatefulPartitionedCall?$block4_conv1/StatefulPartitionedCall?$block4_conv2/StatefulPartitionedCall?$block4_conv3/StatefulPartitionedCall?$block5_conv1/StatefulPartitionedCall?$block5_conv2/StatefulPartitionedCall?$block5_conv3/StatefulPartitionedCall?
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_1block1_conv1_106848block1_conv1_106850*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block1_conv1_layer_call_and_return_conditional_losses_1061252&
$block1_conv1/StatefulPartitionedCall?
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_106853block1_conv2_106855*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block1_conv2_layer_call_and_return_conditional_losses_1061422&
$block1_conv2/StatefulPartitionedCall?
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block1_pool_layer_call_and_return_conditional_losses_1060532
block1_pool/PartitionedCall?
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_106859block2_conv1_106861*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block2_conv1_layer_call_and_return_conditional_losses_1061602&
$block2_conv1/StatefulPartitionedCall?
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_106864block2_conv2_106866*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block2_conv2_layer_call_and_return_conditional_losses_1061772&
$block2_conv2/StatefulPartitionedCall?
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????KK?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block2_pool_layer_call_and_return_conditional_losses_1060652
block2_pool/PartitionedCall?
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_106870block3_conv1_106872*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????KK?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block3_conv1_layer_call_and_return_conditional_losses_1061952&
$block3_conv1/StatefulPartitionedCall?
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_106875block3_conv2_106877*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????KK?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block3_conv2_layer_call_and_return_conditional_losses_1062122&
$block3_conv2/StatefulPartitionedCall?
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_106880block3_conv3_106882*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????KK?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block3_conv3_layer_call_and_return_conditional_losses_1062292&
$block3_conv3/StatefulPartitionedCall?
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????%%?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_pool_layer_call_and_return_conditional_losses_1060772
block3_pool/PartitionedCall?
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_106886block4_conv1_106888*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????%%?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block4_conv1_layer_call_and_return_conditional_losses_1062472&
$block4_conv1/StatefulPartitionedCall?
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_106891block4_conv2_106893*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????%%?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block4_conv2_layer_call_and_return_conditional_losses_1062642&
$block4_conv2/StatefulPartitionedCall?
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_106896block4_conv3_106898*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????%%?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block4_conv3_layer_call_and_return_conditional_losses_1062812&
$block4_conv3/StatefulPartitionedCall?
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_pool_layer_call_and_return_conditional_losses_1060892
block4_pool/PartitionedCall?
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_106902block5_conv1_106904*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block5_conv1_layer_call_and_return_conditional_losses_1062992&
$block5_conv1/StatefulPartitionedCall?
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_106907block5_conv2_106909*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block5_conv2_layer_call_and_return_conditional_losses_1063162&
$block5_conv2/StatefulPartitionedCall?
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_106912block5_conv3_106914*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block5_conv3_layer_call_and_return_conditional_losses_1063332&
$block5_conv3/StatefulPartitionedCall?
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_pool_layer_call_and_return_conditional_losses_1061012
block5_pool/PartitionedCall?
IdentityIdentity$block5_pool/PartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
ؖ
?
A__inference_vgg16_layer_call_and_return_conditional_losses_108885

inputsE
+block1_conv1_conv2d_readvariableop_resource:@:
,block1_conv1_biasadd_readvariableop_resource:@E
+block1_conv2_conv2d_readvariableop_resource:@@:
,block1_conv2_biasadd_readvariableop_resource:@F
+block2_conv1_conv2d_readvariableop_resource:@?;
,block2_conv1_biasadd_readvariableop_resource:	?G
+block2_conv2_conv2d_readvariableop_resource:??;
,block2_conv2_biasadd_readvariableop_resource:	?G
+block3_conv1_conv2d_readvariableop_resource:??;
,block3_conv1_biasadd_readvariableop_resource:	?G
+block3_conv2_conv2d_readvariableop_resource:??;
,block3_conv2_biasadd_readvariableop_resource:	?G
+block3_conv3_conv2d_readvariableop_resource:??;
,block3_conv3_biasadd_readvariableop_resource:	?G
+block4_conv1_conv2d_readvariableop_resource:??;
,block4_conv1_biasadd_readvariableop_resource:	?G
+block4_conv2_conv2d_readvariableop_resource:??;
,block4_conv2_biasadd_readvariableop_resource:	?G
+block4_conv3_conv2d_readvariableop_resource:??;
,block4_conv3_biasadd_readvariableop_resource:	?G
+block5_conv1_conv2d_readvariableop_resource:??;
,block5_conv1_biasadd_readvariableop_resource:	?G
+block5_conv2_conv2d_readvariableop_resource:??;
,block5_conv2_biasadd_readvariableop_resource:	?G
+block5_conv3_conv2d_readvariableop_resource:??;
,block5_conv3_biasadd_readvariableop_resource:	?
identity??#block1_conv1/BiasAdd/ReadVariableOp?"block1_conv1/Conv2D/ReadVariableOp?#block1_conv2/BiasAdd/ReadVariableOp?"block1_conv2/Conv2D/ReadVariableOp?#block2_conv1/BiasAdd/ReadVariableOp?"block2_conv1/Conv2D/ReadVariableOp?#block2_conv2/BiasAdd/ReadVariableOp?"block2_conv2/Conv2D/ReadVariableOp?#block3_conv1/BiasAdd/ReadVariableOp?"block3_conv1/Conv2D/ReadVariableOp?#block3_conv2/BiasAdd/ReadVariableOp?"block3_conv2/Conv2D/ReadVariableOp?#block3_conv3/BiasAdd/ReadVariableOp?"block3_conv3/Conv2D/ReadVariableOp?#block4_conv1/BiasAdd/ReadVariableOp?"block4_conv1/Conv2D/ReadVariableOp?#block4_conv2/BiasAdd/ReadVariableOp?"block4_conv2/Conv2D/ReadVariableOp?#block4_conv3/BiasAdd/ReadVariableOp?"block4_conv3/Conv2D/ReadVariableOp?#block5_conv1/BiasAdd/ReadVariableOp?"block5_conv1/Conv2D/ReadVariableOp?#block5_conv2/BiasAdd/ReadVariableOp?"block5_conv2/Conv2D/ReadVariableOp?#block5_conv3/BiasAdd/ReadVariableOp?"block5_conv3/Conv2D/ReadVariableOp?
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02$
"block1_conv1/Conv2D/ReadVariableOp?
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
block1_conv1/Conv2D?
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv1/BiasAdd/ReadVariableOp?
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
block1_conv1/BiasAdd?
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
block1_conv1/Relu?
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02$
"block1_conv2/Conv2D/ReadVariableOp?
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
block1_conv2/Conv2D?
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv2/BiasAdd/ReadVariableOp?
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
block1_conv2/BiasAdd?
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
block1_conv2/Relu?
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
2
block1_pool/MaxPool?
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02$
"block2_conv1/Conv2D/ReadVariableOp?
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
block2_conv1/Conv2D?
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block2_conv1/BiasAdd/ReadVariableOp?
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
block2_conv1/BiasAdd?
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
block2_conv1/Relu?
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block2_conv2/Conv2D/ReadVariableOp?
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
block2_conv2/Conv2D?
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block2_conv2/BiasAdd/ReadVariableOp?
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
block2_conv2/BiasAdd?
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
block2_conv2/Relu?
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:?????????KK?*
ksize
*
paddingVALID*
strides
2
block2_pool/MaxPool?
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block3_conv1/Conv2D/ReadVariableOp?
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?*
paddingSAME*
strides
2
block3_conv1/Conv2D?
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block3_conv1/BiasAdd/ReadVariableOp?
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?2
block3_conv1/BiasAdd?
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????KK?2
block3_conv1/Relu?
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block3_conv2/Conv2D/ReadVariableOp?
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?*
paddingSAME*
strides
2
block3_conv2/Conv2D?
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block3_conv2/BiasAdd/ReadVariableOp?
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?2
block3_conv2/BiasAdd?
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????KK?2
block3_conv2/Relu?
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block3_conv3/Conv2D/ReadVariableOp?
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?*
paddingSAME*
strides
2
block3_conv3/Conv2D?
#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block3_conv3/BiasAdd/ReadVariableOp?
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?2
block3_conv3/BiasAdd?
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????KK?2
block3_conv3/Relu?
block3_pool/MaxPoolMaxPoolblock3_conv3/Relu:activations:0*0
_output_shapes
:?????????%%?*
ksize
*
paddingVALID*
strides
2
block3_pool/MaxPool?
"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block4_conv1/Conv2D/ReadVariableOp?
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?*
paddingSAME*
strides
2
block4_conv1/Conv2D?
#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block4_conv1/BiasAdd/ReadVariableOp?
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?2
block4_conv1/BiasAdd?
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????%%?2
block4_conv1/Relu?
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block4_conv2/Conv2D/ReadVariableOp?
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?*
paddingSAME*
strides
2
block4_conv2/Conv2D?
#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block4_conv2/BiasAdd/ReadVariableOp?
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?2
block4_conv2/BiasAdd?
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????%%?2
block4_conv2/Relu?
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block4_conv3/Conv2D/ReadVariableOp?
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?*
paddingSAME*
strides
2
block4_conv3/Conv2D?
#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block4_conv3/BiasAdd/ReadVariableOp?
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?2
block4_conv3/BiasAdd?
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????%%?2
block4_conv3/Relu?
block4_pool/MaxPoolMaxPoolblock4_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
block4_pool/MaxPool?
"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block5_conv1/Conv2D/ReadVariableOp?
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
block5_conv1/Conv2D?
#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block5_conv1/BiasAdd/ReadVariableOp?
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
block5_conv1/BiasAdd?
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block5_conv1/Relu?
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block5_conv2/Conv2D/ReadVariableOp?
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
block5_conv2/Conv2D?
#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block5_conv2/BiasAdd/ReadVariableOp?
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
block5_conv2/BiasAdd?
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block5_conv2/Relu?
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block5_conv3/Conv2D/ReadVariableOp?
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
block5_conv3/Conv2D?
#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block5_conv3/BiasAdd/ReadVariableOp?
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
block5_conv3/BiasAdd?
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block5_conv3/Relu?
block5_pool/MaxPoolMaxPoolblock5_conv3/Relu:activations:0*0
_output_shapes
:?????????		?*
ksize
*
paddingVALID*
strides
2
block5_pool/MaxPool?
IdentityIdentityblock5_pool/MaxPool:output:0$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2J
#block5_conv2/BiasAdd/ReadVariableOp#block5_conv2/BiasAdd/ReadVariableOp2H
"block5_conv2/Conv2D/ReadVariableOp"block5_conv2/Conv2D/ReadVariableOp2J
#block5_conv3/BiasAdd/ReadVariableOp#block5_conv3/BiasAdd/ReadVariableOp2H
"block5_conv3/Conv2D/ReadVariableOp"block5_conv3/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
0__inference_cartoon_hidden3_layer_call_fn_109005

inputs
unknown:

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_cartoon_hidden3_layer_call_and_return_conditional_losses_1071042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
J__inference_cartoon_output_layer_call_and_return_conditional_losses_109056

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
H__inference_block4_conv3_layer_call_and_return_conditional_losses_109276

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????%%?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????%%?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????%%?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????%%?
 
_user_specified_nameinputs
?
?
H__inference_block4_conv2_layer_call_and_return_conditional_losses_109256

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????%%?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????%%?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????%%?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????%%?
 
_user_specified_nameinputs
?

?
M__inference_watermark_hidden2_layer_call_and_return_conditional_losses_107019

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
2__inference_watermark_hidden3_layer_call_fn_108985

inputs
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_watermark_hidden3_layer_call_and_return_conditional_losses_1070532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
M__inference_watermark_hidden4_layer_call_and_return_conditional_losses_107087

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
K__inference_cartoon_hidden3_layer_call_and_return_conditional_losses_107104

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
-__inference_block5_conv2_layer_call_fn_109305

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block5_conv2_layer_call_and_return_conditional_losses_1063162
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?\
?
A__inference_vgg16_layer_call_and_return_conditional_losses_106341

inputs-
block1_conv1_106126:@!
block1_conv1_106128:@-
block1_conv2_106143:@@!
block1_conv2_106145:@.
block2_conv1_106161:@?"
block2_conv1_106163:	?/
block2_conv2_106178:??"
block2_conv2_106180:	?/
block3_conv1_106196:??"
block3_conv1_106198:	?/
block3_conv2_106213:??"
block3_conv2_106215:	?/
block3_conv3_106230:??"
block3_conv3_106232:	?/
block4_conv1_106248:??"
block4_conv1_106250:	?/
block4_conv2_106265:??"
block4_conv2_106267:	?/
block4_conv3_106282:??"
block4_conv3_106284:	?/
block5_conv1_106300:??"
block5_conv1_106302:	?/
block5_conv2_106317:??"
block5_conv2_106319:	?/
block5_conv3_106334:??"
block5_conv3_106336:	?
identity??$block1_conv1/StatefulPartitionedCall?$block1_conv2/StatefulPartitionedCall?$block2_conv1/StatefulPartitionedCall?$block2_conv2/StatefulPartitionedCall?$block3_conv1/StatefulPartitionedCall?$block3_conv2/StatefulPartitionedCall?$block3_conv3/StatefulPartitionedCall?$block4_conv1/StatefulPartitionedCall?$block4_conv2/StatefulPartitionedCall?$block4_conv3/StatefulPartitionedCall?$block5_conv1/StatefulPartitionedCall?$block5_conv2/StatefulPartitionedCall?$block5_conv3/StatefulPartitionedCall?
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_106126block1_conv1_106128*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block1_conv1_layer_call_and_return_conditional_losses_1061252&
$block1_conv1/StatefulPartitionedCall?
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_106143block1_conv2_106145*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block1_conv2_layer_call_and_return_conditional_losses_1061422&
$block1_conv2/StatefulPartitionedCall?
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block1_pool_layer_call_and_return_conditional_losses_1060532
block1_pool/PartitionedCall?
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_106161block2_conv1_106163*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block2_conv1_layer_call_and_return_conditional_losses_1061602&
$block2_conv1/StatefulPartitionedCall?
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_106178block2_conv2_106180*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block2_conv2_layer_call_and_return_conditional_losses_1061772&
$block2_conv2/StatefulPartitionedCall?
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????KK?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block2_pool_layer_call_and_return_conditional_losses_1060652
block2_pool/PartitionedCall?
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_106196block3_conv1_106198*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????KK?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block3_conv1_layer_call_and_return_conditional_losses_1061952&
$block3_conv1/StatefulPartitionedCall?
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_106213block3_conv2_106215*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????KK?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block3_conv2_layer_call_and_return_conditional_losses_1062122&
$block3_conv2/StatefulPartitionedCall?
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_106230block3_conv3_106232*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????KK?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block3_conv3_layer_call_and_return_conditional_losses_1062292&
$block3_conv3/StatefulPartitionedCall?
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????%%?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_pool_layer_call_and_return_conditional_losses_1060772
block3_pool/PartitionedCall?
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_106248block4_conv1_106250*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????%%?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block4_conv1_layer_call_and_return_conditional_losses_1062472&
$block4_conv1/StatefulPartitionedCall?
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_106265block4_conv2_106267*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????%%?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block4_conv2_layer_call_and_return_conditional_losses_1062642&
$block4_conv2/StatefulPartitionedCall?
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_106282block4_conv3_106284*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????%%?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block4_conv3_layer_call_and_return_conditional_losses_1062812&
$block4_conv3/StatefulPartitionedCall?
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_pool_layer_call_and_return_conditional_losses_1060892
block4_pool/PartitionedCall?
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_106300block5_conv1_106302*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block5_conv1_layer_call_and_return_conditional_losses_1062992&
$block5_conv1/StatefulPartitionedCall?
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_106317block5_conv2_106319*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block5_conv2_layer_call_and_return_conditional_losses_1063162&
$block5_conv2/StatefulPartitionedCall?
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_106334block5_conv3_106336*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block5_conv3_layer_call_and_return_conditional_losses_1063332&
$block5_conv3/StatefulPartitionedCall?
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_pool_layer_call_and_return_conditional_losses_1061012
block5_pool/PartitionedCall?
IdentityIdentity$block5_pool/PartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
c
G__inference_block5_pool_layer_call_and_return_conditional_losses_106101

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
K__inference_cartoon_hidden3_layer_call_and_return_conditional_losses_109016

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
M__inference_watermark_hidden1_layer_call_and_return_conditional_losses_108916

inputs2
matmul_readvariableop_resource:
??@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
-__inference_block3_conv3_layer_call_fn_109205

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????KK?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block3_conv3_layer_call_and_return_conditional_losses_1062292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????KK?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????KK?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????KK?
 
_user_specified_nameinputs
?
?
H__inference_block1_conv1_layer_call_and_return_conditional_losses_109096

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
-__inference_block2_conv2_layer_call_fn_109145

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block2_conv2_layer_call_and_return_conditional_losses_1061772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
-__inference_block4_conv2_layer_call_fn_109245

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????%%?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block4_conv2_layer_call_and_return_conditional_losses_1062642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????%%?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????%%?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????%%?
 
_user_specified_nameinputs
??
?
"__inference__traced_restore_109658
file_prefix=
)assignvariableop_watermark_hidden1_kernel:
??@7
)assignvariableop_1_watermark_hidden1_bias:@=
)assignvariableop_2_cartoon_hidden1_kernel:
??@5
'assignvariableop_3_cartoon_hidden1_bias:@=
+assignvariableop_4_watermark_hidden2_kernel:@ 7
)assignvariableop_5_watermark_hidden2_bias: ;
)assignvariableop_6_cartoon_hidden2_kernel:@5
'assignvariableop_7_cartoon_hidden2_bias:=
+assignvariableop_8_watermark_hidden3_kernel: 7
)assignvariableop_9_watermark_hidden3_bias:<
*assignvariableop_10_cartoon_hidden3_kernel:
6
(assignvariableop_11_cartoon_hidden3_bias:
>
,assignvariableop_12_watermark_hidden4_kernel:
8
*assignvariableop_13_watermark_hidden4_bias:
;
)assignvariableop_14_cartoon_output_kernel:
5
'assignvariableop_15_cartoon_output_bias:=
+assignvariableop_16_watermark_output_kernel:
7
)assignvariableop_17_watermark_output_bias:A
'assignvariableop_18_block1_conv1_kernel:@3
%assignvariableop_19_block1_conv1_bias:@A
'assignvariableop_20_block1_conv2_kernel:@@3
%assignvariableop_21_block1_conv2_bias:@B
'assignvariableop_22_block2_conv1_kernel:@?4
%assignvariableop_23_block2_conv1_bias:	?C
'assignvariableop_24_block2_conv2_kernel:??4
%assignvariableop_25_block2_conv2_bias:	?C
'assignvariableop_26_block3_conv1_kernel:??4
%assignvariableop_27_block3_conv1_bias:	?C
'assignvariableop_28_block3_conv2_kernel:??4
%assignvariableop_29_block3_conv2_bias:	?C
'assignvariableop_30_block3_conv3_kernel:??4
%assignvariableop_31_block3_conv3_bias:	?C
'assignvariableop_32_block4_conv1_kernel:??4
%assignvariableop_33_block4_conv1_bias:	?C
'assignvariableop_34_block4_conv2_kernel:??4
%assignvariableop_35_block4_conv2_bias:	?C
'assignvariableop_36_block4_conv3_kernel:??4
%assignvariableop_37_block4_conv3_bias:	?C
'assignvariableop_38_block5_conv1_kernel:??4
%assignvariableop_39_block5_conv1_bias:	?C
'assignvariableop_40_block5_conv2_kernel:??4
%assignvariableop_41_block5_conv2_bias:	?C
'assignvariableop_42_block5_conv3_kernel:??4
%assignvariableop_43_block5_conv3_bias:	?#
assignvariableop_44_total: #
assignvariableop_45_count: %
assignvariableop_46_total_1: %
assignvariableop_47_count_1: 
identity_49??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*?
value?B?1B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*u
valuelBj1B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes5
3212
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp)assignvariableop_watermark_hidden1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp)assignvariableop_1_watermark_hidden1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp)assignvariableop_2_cartoon_hidden1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp'assignvariableop_3_cartoon_hidden1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp+assignvariableop_4_watermark_hidden2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp)assignvariableop_5_watermark_hidden2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp)assignvariableop_6_cartoon_hidden2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp'assignvariableop_7_cartoon_hidden2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp+assignvariableop_8_watermark_hidden3_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp)assignvariableop_9_watermark_hidden3_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp*assignvariableop_10_cartoon_hidden3_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp(assignvariableop_11_cartoon_hidden3_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp,assignvariableop_12_watermark_hidden4_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp*assignvariableop_13_watermark_hidden4_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp)assignvariableop_14_cartoon_output_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp'assignvariableop_15_cartoon_output_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp+assignvariableop_16_watermark_output_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp)assignvariableop_17_watermark_output_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp'assignvariableop_18_block1_conv1_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp%assignvariableop_19_block1_conv1_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp'assignvariableop_20_block1_conv2_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp%assignvariableop_21_block1_conv2_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp'assignvariableop_22_block2_conv1_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp%assignvariableop_23_block2_conv1_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp'assignvariableop_24_block2_conv2_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp%assignvariableop_25_block2_conv2_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp'assignvariableop_26_block3_conv1_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp%assignvariableop_27_block3_conv1_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp'assignvariableop_28_block3_conv2_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp%assignvariableop_29_block3_conv2_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp'assignvariableop_30_block3_conv3_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp%assignvariableop_31_block3_conv3_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp'assignvariableop_32_block4_conv1_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp%assignvariableop_33_block4_conv1_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp'assignvariableop_34_block4_conv2_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp%assignvariableop_35_block4_conv2_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp'assignvariableop_36_block4_conv3_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp%assignvariableop_37_block4_conv3_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp'assignvariableop_38_block5_conv1_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp%assignvariableop_39_block5_conv1_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp'assignvariableop_40_block5_conv2_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp%assignvariableop_41_block5_conv2_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp'assignvariableop_42_block5_conv3_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp%assignvariableop_43_block5_conv3_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOpassignvariableop_44_totalIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOpassignvariableop_45_countIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOpassignvariableop_46_total_1Identity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOpassignvariableop_47_count_1Identity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_479
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_48Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_48?
Identity_49IdentityIdentity_48:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_49"#
identity_49Identity_49:output:0*u
_input_shapesd
b: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
1__inference_watermark_output_layer_call_fn_109065

inputs
unknown:

	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_watermark_output_layer_call_and_return_conditional_losses_1071212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
??
?'
D__inference_model_28_layer_call_and_return_conditional_losses_108571

inputsK
1vgg16_block1_conv1_conv2d_readvariableop_resource:@@
2vgg16_block1_conv1_biasadd_readvariableop_resource:@K
1vgg16_block1_conv2_conv2d_readvariableop_resource:@@@
2vgg16_block1_conv2_biasadd_readvariableop_resource:@L
1vgg16_block2_conv1_conv2d_readvariableop_resource:@?A
2vgg16_block2_conv1_biasadd_readvariableop_resource:	?M
1vgg16_block2_conv2_conv2d_readvariableop_resource:??A
2vgg16_block2_conv2_biasadd_readvariableop_resource:	?M
1vgg16_block3_conv1_conv2d_readvariableop_resource:??A
2vgg16_block3_conv1_biasadd_readvariableop_resource:	?M
1vgg16_block3_conv2_conv2d_readvariableop_resource:??A
2vgg16_block3_conv2_biasadd_readvariableop_resource:	?M
1vgg16_block3_conv3_conv2d_readvariableop_resource:??A
2vgg16_block3_conv3_biasadd_readvariableop_resource:	?M
1vgg16_block4_conv1_conv2d_readvariableop_resource:??A
2vgg16_block4_conv1_biasadd_readvariableop_resource:	?M
1vgg16_block4_conv2_conv2d_readvariableop_resource:??A
2vgg16_block4_conv2_biasadd_readvariableop_resource:	?M
1vgg16_block4_conv3_conv2d_readvariableop_resource:??A
2vgg16_block4_conv3_biasadd_readvariableop_resource:	?M
1vgg16_block5_conv1_conv2d_readvariableop_resource:??A
2vgg16_block5_conv1_biasadd_readvariableop_resource:	?M
1vgg16_block5_conv2_conv2d_readvariableop_resource:??A
2vgg16_block5_conv2_biasadd_readvariableop_resource:	?M
1vgg16_block5_conv3_conv2d_readvariableop_resource:??A
2vgg16_block5_conv3_biasadd_readvariableop_resource:	?D
0watermark_hidden1_matmul_readvariableop_resource:
??@?
1watermark_hidden1_biasadd_readvariableop_resource:@B
0watermark_hidden2_matmul_readvariableop_resource:@ ?
1watermark_hidden2_biasadd_readvariableop_resource: B
.cartoon_hidden1_matmul_readvariableop_resource:
??@=
/cartoon_hidden1_biasadd_readvariableop_resource:@B
0watermark_hidden3_matmul_readvariableop_resource: ?
1watermark_hidden3_biasadd_readvariableop_resource:@
.cartoon_hidden2_matmul_readvariableop_resource:@=
/cartoon_hidden2_biasadd_readvariableop_resource:B
0watermark_hidden4_matmul_readvariableop_resource:
?
1watermark_hidden4_biasadd_readvariableop_resource:
@
.cartoon_hidden3_matmul_readvariableop_resource:
=
/cartoon_hidden3_biasadd_readvariableop_resource:
A
/watermark_output_matmul_readvariableop_resource:
>
0watermark_output_biasadd_readvariableop_resource:?
-cartoon_output_matmul_readvariableop_resource:
<
.cartoon_output_biasadd_readvariableop_resource:
identity

identity_1??&cartoon_hidden1/BiasAdd/ReadVariableOp?%cartoon_hidden1/MatMul/ReadVariableOp?&cartoon_hidden2/BiasAdd/ReadVariableOp?%cartoon_hidden2/MatMul/ReadVariableOp?&cartoon_hidden3/BiasAdd/ReadVariableOp?%cartoon_hidden3/MatMul/ReadVariableOp?%cartoon_output/BiasAdd/ReadVariableOp?$cartoon_output/MatMul/ReadVariableOp?)vgg16/block1_conv1/BiasAdd/ReadVariableOp?(vgg16/block1_conv1/Conv2D/ReadVariableOp?)vgg16/block1_conv2/BiasAdd/ReadVariableOp?(vgg16/block1_conv2/Conv2D/ReadVariableOp?)vgg16/block2_conv1/BiasAdd/ReadVariableOp?(vgg16/block2_conv1/Conv2D/ReadVariableOp?)vgg16/block2_conv2/BiasAdd/ReadVariableOp?(vgg16/block2_conv2/Conv2D/ReadVariableOp?)vgg16/block3_conv1/BiasAdd/ReadVariableOp?(vgg16/block3_conv1/Conv2D/ReadVariableOp?)vgg16/block3_conv2/BiasAdd/ReadVariableOp?(vgg16/block3_conv2/Conv2D/ReadVariableOp?)vgg16/block3_conv3/BiasAdd/ReadVariableOp?(vgg16/block3_conv3/Conv2D/ReadVariableOp?)vgg16/block4_conv1/BiasAdd/ReadVariableOp?(vgg16/block4_conv1/Conv2D/ReadVariableOp?)vgg16/block4_conv2/BiasAdd/ReadVariableOp?(vgg16/block4_conv2/Conv2D/ReadVariableOp?)vgg16/block4_conv3/BiasAdd/ReadVariableOp?(vgg16/block4_conv3/Conv2D/ReadVariableOp?)vgg16/block5_conv1/BiasAdd/ReadVariableOp?(vgg16/block5_conv1/Conv2D/ReadVariableOp?)vgg16/block5_conv2/BiasAdd/ReadVariableOp?(vgg16/block5_conv2/Conv2D/ReadVariableOp?)vgg16/block5_conv3/BiasAdd/ReadVariableOp?(vgg16/block5_conv3/Conv2D/ReadVariableOp?(watermark_hidden1/BiasAdd/ReadVariableOp?'watermark_hidden1/MatMul/ReadVariableOp?(watermark_hidden2/BiasAdd/ReadVariableOp?'watermark_hidden2/MatMul/ReadVariableOp?(watermark_hidden3/BiasAdd/ReadVariableOp?'watermark_hidden3/MatMul/ReadVariableOp?(watermark_hidden4/BiasAdd/ReadVariableOp?'watermark_hidden4/MatMul/ReadVariableOp?'watermark_output/BiasAdd/ReadVariableOp?&watermark_output/MatMul/ReadVariableOpo
rescaling_24/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???;2
rescaling_24/Cast/xs
rescaling_24/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling_24/Cast_1/x?
rescaling_24/mulMulinputsrescaling_24/Cast/x:output:0*
T0*1
_output_shapes
:???????????2
rescaling_24/mul?
rescaling_24/addAddV2rescaling_24/mul:z:0rescaling_24/Cast_1/x:output:0*
T0*1
_output_shapes
:???????????2
rescaling_24/add?
(vgg16/block1_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02*
(vgg16/block1_conv1/Conv2D/ReadVariableOp?
vgg16/block1_conv1/Conv2DConv2Drescaling_24/add:z:00vgg16/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
vgg16/block1_conv1/Conv2D?
)vgg16/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)vgg16/block1_conv1/BiasAdd/ReadVariableOp?
vgg16/block1_conv1/BiasAddBiasAdd"vgg16/block1_conv1/Conv2D:output:01vgg16/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
vgg16/block1_conv1/BiasAdd?
vgg16/block1_conv1/ReluRelu#vgg16/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
vgg16/block1_conv1/Relu?
(vgg16/block1_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02*
(vgg16/block1_conv2/Conv2D/ReadVariableOp?
vgg16/block1_conv2/Conv2DConv2D%vgg16/block1_conv1/Relu:activations:00vgg16/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
vgg16/block1_conv2/Conv2D?
)vgg16/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)vgg16/block1_conv2/BiasAdd/ReadVariableOp?
vgg16/block1_conv2/BiasAddBiasAdd"vgg16/block1_conv2/Conv2D:output:01vgg16/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
vgg16/block1_conv2/BiasAdd?
vgg16/block1_conv2/ReluRelu#vgg16/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
vgg16/block1_conv2/Relu?
vgg16/block1_pool/MaxPoolMaxPool%vgg16/block1_conv2/Relu:activations:0*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
2
vgg16/block1_pool/MaxPool?
(vgg16/block2_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02*
(vgg16/block2_conv1/Conv2D/ReadVariableOp?
vgg16/block2_conv1/Conv2DConv2D"vgg16/block1_pool/MaxPool:output:00vgg16/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
vgg16/block2_conv1/Conv2D?
)vgg16/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block2_conv1/BiasAdd/ReadVariableOp?
vgg16/block2_conv1/BiasAddBiasAdd"vgg16/block2_conv1/Conv2D:output:01vgg16/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
vgg16/block2_conv1/BiasAdd?
vgg16/block2_conv1/ReluRelu#vgg16/block2_conv1/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
vgg16/block2_conv1/Relu?
(vgg16/block2_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block2_conv2/Conv2D/ReadVariableOp?
vgg16/block2_conv2/Conv2DConv2D%vgg16/block2_conv1/Relu:activations:00vgg16/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
vgg16/block2_conv2/Conv2D?
)vgg16/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block2_conv2/BiasAdd/ReadVariableOp?
vgg16/block2_conv2/BiasAddBiasAdd"vgg16/block2_conv2/Conv2D:output:01vgg16/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
vgg16/block2_conv2/BiasAdd?
vgg16/block2_conv2/ReluRelu#vgg16/block2_conv2/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
vgg16/block2_conv2/Relu?
vgg16/block2_pool/MaxPoolMaxPool%vgg16/block2_conv2/Relu:activations:0*0
_output_shapes
:?????????KK?*
ksize
*
paddingVALID*
strides
2
vgg16/block2_pool/MaxPool?
(vgg16/block3_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block3_conv1/Conv2D/ReadVariableOp?
vgg16/block3_conv1/Conv2DConv2D"vgg16/block2_pool/MaxPool:output:00vgg16/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?*
paddingSAME*
strides
2
vgg16/block3_conv1/Conv2D?
)vgg16/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block3_conv1/BiasAdd/ReadVariableOp?
vgg16/block3_conv1/BiasAddBiasAdd"vgg16/block3_conv1/Conv2D:output:01vgg16/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?2
vgg16/block3_conv1/BiasAdd?
vgg16/block3_conv1/ReluRelu#vgg16/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????KK?2
vgg16/block3_conv1/Relu?
(vgg16/block3_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block3_conv2/Conv2D/ReadVariableOp?
vgg16/block3_conv2/Conv2DConv2D%vgg16/block3_conv1/Relu:activations:00vgg16/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?*
paddingSAME*
strides
2
vgg16/block3_conv2/Conv2D?
)vgg16/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block3_conv2/BiasAdd/ReadVariableOp?
vgg16/block3_conv2/BiasAddBiasAdd"vgg16/block3_conv2/Conv2D:output:01vgg16/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?2
vgg16/block3_conv2/BiasAdd?
vgg16/block3_conv2/ReluRelu#vgg16/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????KK?2
vgg16/block3_conv2/Relu?
(vgg16/block3_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg16_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block3_conv3/Conv2D/ReadVariableOp?
vgg16/block3_conv3/Conv2DConv2D%vgg16/block3_conv2/Relu:activations:00vgg16/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?*
paddingSAME*
strides
2
vgg16/block3_conv3/Conv2D?
)vgg16/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block3_conv3/BiasAdd/ReadVariableOp?
vgg16/block3_conv3/BiasAddBiasAdd"vgg16/block3_conv3/Conv2D:output:01vgg16/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?2
vgg16/block3_conv3/BiasAdd?
vgg16/block3_conv3/ReluRelu#vgg16/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????KK?2
vgg16/block3_conv3/Relu?
vgg16/block3_pool/MaxPoolMaxPool%vgg16/block3_conv3/Relu:activations:0*0
_output_shapes
:?????????%%?*
ksize
*
paddingVALID*
strides
2
vgg16/block3_pool/MaxPool?
(vgg16/block4_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block4_conv1/Conv2D/ReadVariableOp?
vgg16/block4_conv1/Conv2DConv2D"vgg16/block3_pool/MaxPool:output:00vgg16/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?*
paddingSAME*
strides
2
vgg16/block4_conv1/Conv2D?
)vgg16/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block4_conv1/BiasAdd/ReadVariableOp?
vgg16/block4_conv1/BiasAddBiasAdd"vgg16/block4_conv1/Conv2D:output:01vgg16/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?2
vgg16/block4_conv1/BiasAdd?
vgg16/block4_conv1/ReluRelu#vgg16/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????%%?2
vgg16/block4_conv1/Relu?
(vgg16/block4_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block4_conv2/Conv2D/ReadVariableOp?
vgg16/block4_conv2/Conv2DConv2D%vgg16/block4_conv1/Relu:activations:00vgg16/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?*
paddingSAME*
strides
2
vgg16/block4_conv2/Conv2D?
)vgg16/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block4_conv2/BiasAdd/ReadVariableOp?
vgg16/block4_conv2/BiasAddBiasAdd"vgg16/block4_conv2/Conv2D:output:01vgg16/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?2
vgg16/block4_conv2/BiasAdd?
vgg16/block4_conv2/ReluRelu#vgg16/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????%%?2
vgg16/block4_conv2/Relu?
(vgg16/block4_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg16_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block4_conv3/Conv2D/ReadVariableOp?
vgg16/block4_conv3/Conv2DConv2D%vgg16/block4_conv2/Relu:activations:00vgg16/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?*
paddingSAME*
strides
2
vgg16/block4_conv3/Conv2D?
)vgg16/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block4_conv3/BiasAdd/ReadVariableOp?
vgg16/block4_conv3/BiasAddBiasAdd"vgg16/block4_conv3/Conv2D:output:01vgg16/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?2
vgg16/block4_conv3/BiasAdd?
vgg16/block4_conv3/ReluRelu#vgg16/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????%%?2
vgg16/block4_conv3/Relu?
vgg16/block4_pool/MaxPoolMaxPool%vgg16/block4_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
vgg16/block4_pool/MaxPool?
(vgg16/block5_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block5_conv1/Conv2D/ReadVariableOp?
vgg16/block5_conv1/Conv2DConv2D"vgg16/block4_pool/MaxPool:output:00vgg16/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
vgg16/block5_conv1/Conv2D?
)vgg16/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block5_conv1/BiasAdd/ReadVariableOp?
vgg16/block5_conv1/BiasAddBiasAdd"vgg16/block5_conv1/Conv2D:output:01vgg16/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
vgg16/block5_conv1/BiasAdd?
vgg16/block5_conv1/ReluRelu#vgg16/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
vgg16/block5_conv1/Relu?
(vgg16/block5_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block5_conv2/Conv2D/ReadVariableOp?
vgg16/block5_conv2/Conv2DConv2D%vgg16/block5_conv1/Relu:activations:00vgg16/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
vgg16/block5_conv2/Conv2D?
)vgg16/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block5_conv2/BiasAdd/ReadVariableOp?
vgg16/block5_conv2/BiasAddBiasAdd"vgg16/block5_conv2/Conv2D:output:01vgg16/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
vgg16/block5_conv2/BiasAdd?
vgg16/block5_conv2/ReluRelu#vgg16/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
vgg16/block5_conv2/Relu?
(vgg16/block5_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg16_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block5_conv3/Conv2D/ReadVariableOp?
vgg16/block5_conv3/Conv2DConv2D%vgg16/block5_conv2/Relu:activations:00vgg16/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
vgg16/block5_conv3/Conv2D?
)vgg16/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block5_conv3/BiasAdd/ReadVariableOp?
vgg16/block5_conv3/BiasAddBiasAdd"vgg16/block5_conv3/Conv2D:output:01vgg16/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
vgg16/block5_conv3/BiasAdd?
vgg16/block5_conv3/ReluRelu#vgg16/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
vgg16/block5_conv3/Relu?
vgg16/block5_pool/MaxPoolMaxPool%vgg16/block5_conv3/Relu:activations:0*0
_output_shapes
:?????????		?*
ksize
*
paddingVALID*
strides
2
vgg16/block5_pool/MaxPoolu
flatten_28/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ?  2
flatten_28/Const?
flatten_28/ReshapeReshape"vgg16/block5_pool/MaxPool:output:0flatten_28/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_28/Reshape?
'watermark_hidden1/MatMul/ReadVariableOpReadVariableOp0watermark_hidden1_matmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype02)
'watermark_hidden1/MatMul/ReadVariableOp?
watermark_hidden1/MatMulMatMulflatten_28/Reshape:output:0/watermark_hidden1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
watermark_hidden1/MatMul?
(watermark_hidden1/BiasAdd/ReadVariableOpReadVariableOp1watermark_hidden1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(watermark_hidden1/BiasAdd/ReadVariableOp?
watermark_hidden1/BiasAddBiasAdd"watermark_hidden1/MatMul:product:00watermark_hidden1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
watermark_hidden1/BiasAdd?
watermark_hidden1/ReluRelu"watermark_hidden1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
watermark_hidden1/Relu?
'watermark_hidden2/MatMul/ReadVariableOpReadVariableOp0watermark_hidden2_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02)
'watermark_hidden2/MatMul/ReadVariableOp?
watermark_hidden2/MatMulMatMul$watermark_hidden1/Relu:activations:0/watermark_hidden2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
watermark_hidden2/MatMul?
(watermark_hidden2/BiasAdd/ReadVariableOpReadVariableOp1watermark_hidden2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(watermark_hidden2/BiasAdd/ReadVariableOp?
watermark_hidden2/BiasAddBiasAdd"watermark_hidden2/MatMul:product:00watermark_hidden2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
watermark_hidden2/BiasAdd?
watermark_hidden2/ReluRelu"watermark_hidden2/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
watermark_hidden2/Relu?
%cartoon_hidden1/MatMul/ReadVariableOpReadVariableOp.cartoon_hidden1_matmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype02'
%cartoon_hidden1/MatMul/ReadVariableOp?
cartoon_hidden1/MatMulMatMulflatten_28/Reshape:output:0-cartoon_hidden1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
cartoon_hidden1/MatMul?
&cartoon_hidden1/BiasAdd/ReadVariableOpReadVariableOp/cartoon_hidden1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&cartoon_hidden1/BiasAdd/ReadVariableOp?
cartoon_hidden1/BiasAddBiasAdd cartoon_hidden1/MatMul:product:0.cartoon_hidden1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
cartoon_hidden1/BiasAdd?
cartoon_hidden1/ReluRelu cartoon_hidden1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
cartoon_hidden1/Relu?
'watermark_hidden3/MatMul/ReadVariableOpReadVariableOp0watermark_hidden3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02)
'watermark_hidden3/MatMul/ReadVariableOp?
watermark_hidden3/MatMulMatMul$watermark_hidden2/Relu:activations:0/watermark_hidden3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
watermark_hidden3/MatMul?
(watermark_hidden3/BiasAdd/ReadVariableOpReadVariableOp1watermark_hidden3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(watermark_hidden3/BiasAdd/ReadVariableOp?
watermark_hidden3/BiasAddBiasAdd"watermark_hidden3/MatMul:product:00watermark_hidden3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
watermark_hidden3/BiasAdd?
watermark_hidden3/ReluRelu"watermark_hidden3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
watermark_hidden3/Relu?
%cartoon_hidden2/MatMul/ReadVariableOpReadVariableOp.cartoon_hidden2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02'
%cartoon_hidden2/MatMul/ReadVariableOp?
cartoon_hidden2/MatMulMatMul"cartoon_hidden1/Relu:activations:0-cartoon_hidden2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
cartoon_hidden2/MatMul?
&cartoon_hidden2/BiasAdd/ReadVariableOpReadVariableOp/cartoon_hidden2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&cartoon_hidden2/BiasAdd/ReadVariableOp?
cartoon_hidden2/BiasAddBiasAdd cartoon_hidden2/MatMul:product:0.cartoon_hidden2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
cartoon_hidden2/BiasAdd?
cartoon_hidden2/ReluRelu cartoon_hidden2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
cartoon_hidden2/Relu?
'watermark_hidden4/MatMul/ReadVariableOpReadVariableOp0watermark_hidden4_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02)
'watermark_hidden4/MatMul/ReadVariableOp?
watermark_hidden4/MatMulMatMul$watermark_hidden3/Relu:activations:0/watermark_hidden4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
watermark_hidden4/MatMul?
(watermark_hidden4/BiasAdd/ReadVariableOpReadVariableOp1watermark_hidden4_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02*
(watermark_hidden4/BiasAdd/ReadVariableOp?
watermark_hidden4/BiasAddBiasAdd"watermark_hidden4/MatMul:product:00watermark_hidden4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
watermark_hidden4/BiasAdd?
watermark_hidden4/ReluRelu"watermark_hidden4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
watermark_hidden4/Relu?
%cartoon_hidden3/MatMul/ReadVariableOpReadVariableOp.cartoon_hidden3_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02'
%cartoon_hidden3/MatMul/ReadVariableOp?
cartoon_hidden3/MatMulMatMul"cartoon_hidden2/Relu:activations:0-cartoon_hidden3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
cartoon_hidden3/MatMul?
&cartoon_hidden3/BiasAdd/ReadVariableOpReadVariableOp/cartoon_hidden3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02(
&cartoon_hidden3/BiasAdd/ReadVariableOp?
cartoon_hidden3/BiasAddBiasAdd cartoon_hidden3/MatMul:product:0.cartoon_hidden3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
cartoon_hidden3/BiasAdd?
cartoon_hidden3/ReluRelu cartoon_hidden3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
cartoon_hidden3/Relu?
&watermark_output/MatMul/ReadVariableOpReadVariableOp/watermark_output_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02(
&watermark_output/MatMul/ReadVariableOp?
watermark_output/MatMulMatMul$watermark_hidden4/Relu:activations:0.watermark_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
watermark_output/MatMul?
'watermark_output/BiasAdd/ReadVariableOpReadVariableOp0watermark_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'watermark_output/BiasAdd/ReadVariableOp?
watermark_output/BiasAddBiasAdd!watermark_output/MatMul:product:0/watermark_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
watermark_output/BiasAdd?
watermark_output/SigmoidSigmoid!watermark_output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
watermark_output/Sigmoid?
$cartoon_output/MatMul/ReadVariableOpReadVariableOp-cartoon_output_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02&
$cartoon_output/MatMul/ReadVariableOp?
cartoon_output/MatMulMatMul"cartoon_hidden3/Relu:activations:0,cartoon_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
cartoon_output/MatMul?
%cartoon_output/BiasAdd/ReadVariableOpReadVariableOp.cartoon_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%cartoon_output/BiasAdd/ReadVariableOp?
cartoon_output/BiasAddBiasAddcartoon_output/MatMul:product:0-cartoon_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
cartoon_output/BiasAdd?
cartoon_output/SigmoidSigmoidcartoon_output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
cartoon_output/Sigmoid?
IdentityIdentitycartoon_output/Sigmoid:y:0'^cartoon_hidden1/BiasAdd/ReadVariableOp&^cartoon_hidden1/MatMul/ReadVariableOp'^cartoon_hidden2/BiasAdd/ReadVariableOp&^cartoon_hidden2/MatMul/ReadVariableOp'^cartoon_hidden3/BiasAdd/ReadVariableOp&^cartoon_hidden3/MatMul/ReadVariableOp&^cartoon_output/BiasAdd/ReadVariableOp%^cartoon_output/MatMul/ReadVariableOp*^vgg16/block1_conv1/BiasAdd/ReadVariableOp)^vgg16/block1_conv1/Conv2D/ReadVariableOp*^vgg16/block1_conv2/BiasAdd/ReadVariableOp)^vgg16/block1_conv2/Conv2D/ReadVariableOp*^vgg16/block2_conv1/BiasAdd/ReadVariableOp)^vgg16/block2_conv1/Conv2D/ReadVariableOp*^vgg16/block2_conv2/BiasAdd/ReadVariableOp)^vgg16/block2_conv2/Conv2D/ReadVariableOp*^vgg16/block3_conv1/BiasAdd/ReadVariableOp)^vgg16/block3_conv1/Conv2D/ReadVariableOp*^vgg16/block3_conv2/BiasAdd/ReadVariableOp)^vgg16/block3_conv2/Conv2D/ReadVariableOp*^vgg16/block3_conv3/BiasAdd/ReadVariableOp)^vgg16/block3_conv3/Conv2D/ReadVariableOp*^vgg16/block4_conv1/BiasAdd/ReadVariableOp)^vgg16/block4_conv1/Conv2D/ReadVariableOp*^vgg16/block4_conv2/BiasAdd/ReadVariableOp)^vgg16/block4_conv2/Conv2D/ReadVariableOp*^vgg16/block4_conv3/BiasAdd/ReadVariableOp)^vgg16/block4_conv3/Conv2D/ReadVariableOp*^vgg16/block5_conv1/BiasAdd/ReadVariableOp)^vgg16/block5_conv1/Conv2D/ReadVariableOp*^vgg16/block5_conv2/BiasAdd/ReadVariableOp)^vgg16/block5_conv2/Conv2D/ReadVariableOp*^vgg16/block5_conv3/BiasAdd/ReadVariableOp)^vgg16/block5_conv3/Conv2D/ReadVariableOp)^watermark_hidden1/BiasAdd/ReadVariableOp(^watermark_hidden1/MatMul/ReadVariableOp)^watermark_hidden2/BiasAdd/ReadVariableOp(^watermark_hidden2/MatMul/ReadVariableOp)^watermark_hidden3/BiasAdd/ReadVariableOp(^watermark_hidden3/MatMul/ReadVariableOp)^watermark_hidden4/BiasAdd/ReadVariableOp(^watermark_hidden4/MatMul/ReadVariableOp(^watermark_output/BiasAdd/ReadVariableOp'^watermark_output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitywatermark_output/Sigmoid:y:0'^cartoon_hidden1/BiasAdd/ReadVariableOp&^cartoon_hidden1/MatMul/ReadVariableOp'^cartoon_hidden2/BiasAdd/ReadVariableOp&^cartoon_hidden2/MatMul/ReadVariableOp'^cartoon_hidden3/BiasAdd/ReadVariableOp&^cartoon_hidden3/MatMul/ReadVariableOp&^cartoon_output/BiasAdd/ReadVariableOp%^cartoon_output/MatMul/ReadVariableOp*^vgg16/block1_conv1/BiasAdd/ReadVariableOp)^vgg16/block1_conv1/Conv2D/ReadVariableOp*^vgg16/block1_conv2/BiasAdd/ReadVariableOp)^vgg16/block1_conv2/Conv2D/ReadVariableOp*^vgg16/block2_conv1/BiasAdd/ReadVariableOp)^vgg16/block2_conv1/Conv2D/ReadVariableOp*^vgg16/block2_conv2/BiasAdd/ReadVariableOp)^vgg16/block2_conv2/Conv2D/ReadVariableOp*^vgg16/block3_conv1/BiasAdd/ReadVariableOp)^vgg16/block3_conv1/Conv2D/ReadVariableOp*^vgg16/block3_conv2/BiasAdd/ReadVariableOp)^vgg16/block3_conv2/Conv2D/ReadVariableOp*^vgg16/block3_conv3/BiasAdd/ReadVariableOp)^vgg16/block3_conv3/Conv2D/ReadVariableOp*^vgg16/block4_conv1/BiasAdd/ReadVariableOp)^vgg16/block4_conv1/Conv2D/ReadVariableOp*^vgg16/block4_conv2/BiasAdd/ReadVariableOp)^vgg16/block4_conv2/Conv2D/ReadVariableOp*^vgg16/block4_conv3/BiasAdd/ReadVariableOp)^vgg16/block4_conv3/Conv2D/ReadVariableOp*^vgg16/block5_conv1/BiasAdd/ReadVariableOp)^vgg16/block5_conv1/Conv2D/ReadVariableOp*^vgg16/block5_conv2/BiasAdd/ReadVariableOp)^vgg16/block5_conv2/Conv2D/ReadVariableOp*^vgg16/block5_conv3/BiasAdd/ReadVariableOp)^vgg16/block5_conv3/Conv2D/ReadVariableOp)^watermark_hidden1/BiasAdd/ReadVariableOp(^watermark_hidden1/MatMul/ReadVariableOp)^watermark_hidden2/BiasAdd/ReadVariableOp(^watermark_hidden2/MatMul/ReadVariableOp)^watermark_hidden3/BiasAdd/ReadVariableOp(^watermark_hidden3/MatMul/ReadVariableOp)^watermark_hidden4/BiasAdd/ReadVariableOp(^watermark_hidden4/MatMul/ReadVariableOp(^watermark_output/BiasAdd/ReadVariableOp'^watermark_output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesw
u:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&cartoon_hidden1/BiasAdd/ReadVariableOp&cartoon_hidden1/BiasAdd/ReadVariableOp2N
%cartoon_hidden1/MatMul/ReadVariableOp%cartoon_hidden1/MatMul/ReadVariableOp2P
&cartoon_hidden2/BiasAdd/ReadVariableOp&cartoon_hidden2/BiasAdd/ReadVariableOp2N
%cartoon_hidden2/MatMul/ReadVariableOp%cartoon_hidden2/MatMul/ReadVariableOp2P
&cartoon_hidden3/BiasAdd/ReadVariableOp&cartoon_hidden3/BiasAdd/ReadVariableOp2N
%cartoon_hidden3/MatMul/ReadVariableOp%cartoon_hidden3/MatMul/ReadVariableOp2N
%cartoon_output/BiasAdd/ReadVariableOp%cartoon_output/BiasAdd/ReadVariableOp2L
$cartoon_output/MatMul/ReadVariableOp$cartoon_output/MatMul/ReadVariableOp2V
)vgg16/block1_conv1/BiasAdd/ReadVariableOp)vgg16/block1_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block1_conv1/Conv2D/ReadVariableOp(vgg16/block1_conv1/Conv2D/ReadVariableOp2V
)vgg16/block1_conv2/BiasAdd/ReadVariableOp)vgg16/block1_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block1_conv2/Conv2D/ReadVariableOp(vgg16/block1_conv2/Conv2D/ReadVariableOp2V
)vgg16/block2_conv1/BiasAdd/ReadVariableOp)vgg16/block2_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block2_conv1/Conv2D/ReadVariableOp(vgg16/block2_conv1/Conv2D/ReadVariableOp2V
)vgg16/block2_conv2/BiasAdd/ReadVariableOp)vgg16/block2_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block2_conv2/Conv2D/ReadVariableOp(vgg16/block2_conv2/Conv2D/ReadVariableOp2V
)vgg16/block3_conv1/BiasAdd/ReadVariableOp)vgg16/block3_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block3_conv1/Conv2D/ReadVariableOp(vgg16/block3_conv1/Conv2D/ReadVariableOp2V
)vgg16/block3_conv2/BiasAdd/ReadVariableOp)vgg16/block3_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block3_conv2/Conv2D/ReadVariableOp(vgg16/block3_conv2/Conv2D/ReadVariableOp2V
)vgg16/block3_conv3/BiasAdd/ReadVariableOp)vgg16/block3_conv3/BiasAdd/ReadVariableOp2T
(vgg16/block3_conv3/Conv2D/ReadVariableOp(vgg16/block3_conv3/Conv2D/ReadVariableOp2V
)vgg16/block4_conv1/BiasAdd/ReadVariableOp)vgg16/block4_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block4_conv1/Conv2D/ReadVariableOp(vgg16/block4_conv1/Conv2D/ReadVariableOp2V
)vgg16/block4_conv2/BiasAdd/ReadVariableOp)vgg16/block4_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block4_conv2/Conv2D/ReadVariableOp(vgg16/block4_conv2/Conv2D/ReadVariableOp2V
)vgg16/block4_conv3/BiasAdd/ReadVariableOp)vgg16/block4_conv3/BiasAdd/ReadVariableOp2T
(vgg16/block4_conv3/Conv2D/ReadVariableOp(vgg16/block4_conv3/Conv2D/ReadVariableOp2V
)vgg16/block5_conv1/BiasAdd/ReadVariableOp)vgg16/block5_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block5_conv1/Conv2D/ReadVariableOp(vgg16/block5_conv1/Conv2D/ReadVariableOp2V
)vgg16/block5_conv2/BiasAdd/ReadVariableOp)vgg16/block5_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block5_conv2/Conv2D/ReadVariableOp(vgg16/block5_conv2/Conv2D/ReadVariableOp2V
)vgg16/block5_conv3/BiasAdd/ReadVariableOp)vgg16/block5_conv3/BiasAdd/ReadVariableOp2T
(vgg16/block5_conv3/Conv2D/ReadVariableOp(vgg16/block5_conv3/Conv2D/ReadVariableOp2T
(watermark_hidden1/BiasAdd/ReadVariableOp(watermark_hidden1/BiasAdd/ReadVariableOp2R
'watermark_hidden1/MatMul/ReadVariableOp'watermark_hidden1/MatMul/ReadVariableOp2T
(watermark_hidden2/BiasAdd/ReadVariableOp(watermark_hidden2/BiasAdd/ReadVariableOp2R
'watermark_hidden2/MatMul/ReadVariableOp'watermark_hidden2/MatMul/ReadVariableOp2T
(watermark_hidden3/BiasAdd/ReadVariableOp(watermark_hidden3/BiasAdd/ReadVariableOp2R
'watermark_hidden3/MatMul/ReadVariableOp'watermark_hidden3/MatMul/ReadVariableOp2T
(watermark_hidden4/BiasAdd/ReadVariableOp(watermark_hidden4/BiasAdd/ReadVariableOp2R
'watermark_hidden4/MatMul/ReadVariableOp'watermark_hidden4/MatMul/ReadVariableOp2R
'watermark_output/BiasAdd/ReadVariableOp'watermark_output/BiasAdd/ReadVariableOp2P
&watermark_output/MatMul/ReadVariableOp&watermark_output/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?R
?
D__inference_model_28_layer_call_and_return_conditional_losses_107146

inputs&
vgg16_106930:@
vgg16_106932:@&
vgg16_106934:@@
vgg16_106936:@'
vgg16_106938:@?
vgg16_106940:	?(
vgg16_106942:??
vgg16_106944:	?(
vgg16_106946:??
vgg16_106948:	?(
vgg16_106950:??
vgg16_106952:	?(
vgg16_106954:??
vgg16_106956:	?(
vgg16_106958:??
vgg16_106960:	?(
vgg16_106962:??
vgg16_106964:	?(
vgg16_106966:??
vgg16_106968:	?(
vgg16_106970:??
vgg16_106972:	?(
vgg16_106974:??
vgg16_106976:	?(
vgg16_106978:??
vgg16_106980:	?,
watermark_hidden1_107003:
??@&
watermark_hidden1_107005:@*
watermark_hidden2_107020:@ &
watermark_hidden2_107022: *
cartoon_hidden1_107037:
??@$
cartoon_hidden1_107039:@*
watermark_hidden3_107054: &
watermark_hidden3_107056:(
cartoon_hidden2_107071:@$
cartoon_hidden2_107073:*
watermark_hidden4_107088:
&
watermark_hidden4_107090:
(
cartoon_hidden3_107105:
$
cartoon_hidden3_107107:
)
watermark_output_107122:
%
watermark_output_107124:'
cartoon_output_107139:
#
cartoon_output_107141:
identity

identity_1??'cartoon_hidden1/StatefulPartitionedCall?'cartoon_hidden2/StatefulPartitionedCall?'cartoon_hidden3/StatefulPartitionedCall?&cartoon_output/StatefulPartitionedCall?vgg16/StatefulPartitionedCall?)watermark_hidden1/StatefulPartitionedCall?)watermark_hidden2/StatefulPartitionedCall?)watermark_hidden3/StatefulPartitionedCall?)watermark_hidden4/StatefulPartitionedCall?(watermark_output/StatefulPartitionedCallo
rescaling_24/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???;2
rescaling_24/Cast/xs
rescaling_24/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling_24/Cast_1/x?
rescaling_24/mulMulinputsrescaling_24/Cast/x:output:0*
T0*1
_output_shapes
:???????????2
rescaling_24/mul?
rescaling_24/addAddV2rescaling_24/mul:z:0rescaling_24/Cast_1/x:output:0*
T0*1
_output_shapes
:???????????2
rescaling_24/add?
vgg16/StatefulPartitionedCallStatefulPartitionedCallrescaling_24/add:z:0vgg16_106930vgg16_106932vgg16_106934vgg16_106936vgg16_106938vgg16_106940vgg16_106942vgg16_106944vgg16_106946vgg16_106948vgg16_106950vgg16_106952vgg16_106954vgg16_106956vgg16_106958vgg16_106960vgg16_106962vgg16_106964vgg16_106966vgg16_106968vgg16_106970vgg16_106972vgg16_106974vgg16_106976vgg16_106978vgg16_106980*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_vgg16_layer_call_and_return_conditional_losses_1063412
vgg16/StatefulPartitionedCall?
flatten_28/PartitionedCallPartitionedCall&vgg16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_28_layer_call_and_return_conditional_losses_1069892
flatten_28/PartitionedCall?
)watermark_hidden1/StatefulPartitionedCallStatefulPartitionedCall#flatten_28/PartitionedCall:output:0watermark_hidden1_107003watermark_hidden1_107005*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_watermark_hidden1_layer_call_and_return_conditional_losses_1070022+
)watermark_hidden1/StatefulPartitionedCall?
)watermark_hidden2/StatefulPartitionedCallStatefulPartitionedCall2watermark_hidden1/StatefulPartitionedCall:output:0watermark_hidden2_107020watermark_hidden2_107022*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_watermark_hidden2_layer_call_and_return_conditional_losses_1070192+
)watermark_hidden2/StatefulPartitionedCall?
'cartoon_hidden1/StatefulPartitionedCallStatefulPartitionedCall#flatten_28/PartitionedCall:output:0cartoon_hidden1_107037cartoon_hidden1_107039*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_cartoon_hidden1_layer_call_and_return_conditional_losses_1070362)
'cartoon_hidden1/StatefulPartitionedCall?
)watermark_hidden3/StatefulPartitionedCallStatefulPartitionedCall2watermark_hidden2/StatefulPartitionedCall:output:0watermark_hidden3_107054watermark_hidden3_107056*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_watermark_hidden3_layer_call_and_return_conditional_losses_1070532+
)watermark_hidden3/StatefulPartitionedCall?
'cartoon_hidden2/StatefulPartitionedCallStatefulPartitionedCall0cartoon_hidden1/StatefulPartitionedCall:output:0cartoon_hidden2_107071cartoon_hidden2_107073*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_cartoon_hidden2_layer_call_and_return_conditional_losses_1070702)
'cartoon_hidden2/StatefulPartitionedCall?
)watermark_hidden4/StatefulPartitionedCallStatefulPartitionedCall2watermark_hidden3/StatefulPartitionedCall:output:0watermark_hidden4_107088watermark_hidden4_107090*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_watermark_hidden4_layer_call_and_return_conditional_losses_1070872+
)watermark_hidden4/StatefulPartitionedCall?
'cartoon_hidden3/StatefulPartitionedCallStatefulPartitionedCall0cartoon_hidden2/StatefulPartitionedCall:output:0cartoon_hidden3_107105cartoon_hidden3_107107*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_cartoon_hidden3_layer_call_and_return_conditional_losses_1071042)
'cartoon_hidden3/StatefulPartitionedCall?
(watermark_output/StatefulPartitionedCallStatefulPartitionedCall2watermark_hidden4/StatefulPartitionedCall:output:0watermark_output_107122watermark_output_107124*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_watermark_output_layer_call_and_return_conditional_losses_1071212*
(watermark_output/StatefulPartitionedCall?
&cartoon_output/StatefulPartitionedCallStatefulPartitionedCall0cartoon_hidden3/StatefulPartitionedCall:output:0cartoon_output_107139cartoon_output_107141*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_cartoon_output_layer_call_and_return_conditional_losses_1071382(
&cartoon_output/StatefulPartitionedCall?
IdentityIdentity/cartoon_output/StatefulPartitionedCall:output:0(^cartoon_hidden1/StatefulPartitionedCall(^cartoon_hidden2/StatefulPartitionedCall(^cartoon_hidden3/StatefulPartitionedCall'^cartoon_output/StatefulPartitionedCall^vgg16/StatefulPartitionedCall*^watermark_hidden1/StatefulPartitionedCall*^watermark_hidden2/StatefulPartitionedCall*^watermark_hidden3/StatefulPartitionedCall*^watermark_hidden4/StatefulPartitionedCall)^watermark_output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity1watermark_output/StatefulPartitionedCall:output:0(^cartoon_hidden1/StatefulPartitionedCall(^cartoon_hidden2/StatefulPartitionedCall(^cartoon_hidden3/StatefulPartitionedCall'^cartoon_output/StatefulPartitionedCall^vgg16/StatefulPartitionedCall*^watermark_hidden1/StatefulPartitionedCall*^watermark_hidden2/StatefulPartitionedCall*^watermark_hidden3/StatefulPartitionedCall*^watermark_hidden4/StatefulPartitionedCall)^watermark_output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesw
u:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2R
'cartoon_hidden1/StatefulPartitionedCall'cartoon_hidden1/StatefulPartitionedCall2R
'cartoon_hidden2/StatefulPartitionedCall'cartoon_hidden2/StatefulPartitionedCall2R
'cartoon_hidden3/StatefulPartitionedCall'cartoon_hidden3/StatefulPartitionedCall2P
&cartoon_output/StatefulPartitionedCall&cartoon_output/StatefulPartitionedCall2>
vgg16/StatefulPartitionedCallvgg16/StatefulPartitionedCall2V
)watermark_hidden1/StatefulPartitionedCall)watermark_hidden1/StatefulPartitionedCall2V
)watermark_hidden2/StatefulPartitionedCall)watermark_hidden2/StatefulPartitionedCall2V
)watermark_hidden3/StatefulPartitionedCall)watermark_hidden3/StatefulPartitionedCall2V
)watermark_hidden4/StatefulPartitionedCall)watermark_hidden4/StatefulPartitionedCall2T
(watermark_output/StatefulPartitionedCall(watermark_output/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
H__inference_block3_conv3_layer_call_and_return_conditional_losses_106229

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????KK?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????KK?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????KK?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????KK?
 
_user_specified_nameinputs
?
?
&__inference_vgg16_layer_call_fn_108685

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_vgg16_layer_call_and_return_conditional_losses_1066592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
H__inference_block3_conv3_layer_call_and_return_conditional_losses_109216

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????KK?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????KK?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????KK?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????KK?
 
_user_specified_nameinputs
?
?
H__inference_block5_conv2_layer_call_and_return_conditional_losses_109316

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
G__inference_block2_pool_layer_call_and_return_conditional_losses_106065

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
H__inference_block5_conv3_layer_call_and_return_conditional_losses_106333

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
H__inference_block5_conv1_layer_call_and_return_conditional_losses_109296

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
H__inference_block3_conv1_layer_call_and_return_conditional_losses_106195

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????KK?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????KK?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????KK?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????KK?
 
_user_specified_nameinputs
?
?
2__inference_watermark_hidden4_layer_call_fn_109025

inputs
unknown:

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_watermark_hidden4_layer_call_and_return_conditional_losses_1070872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
H
,__inference_block5_pool_layer_call_fn_106107

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_pool_layer_call_and_return_conditional_losses_1061012
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?\
?
__inference__traced_save_109504
file_prefix7
3savev2_watermark_hidden1_kernel_read_readvariableop5
1savev2_watermark_hidden1_bias_read_readvariableop5
1savev2_cartoon_hidden1_kernel_read_readvariableop3
/savev2_cartoon_hidden1_bias_read_readvariableop7
3savev2_watermark_hidden2_kernel_read_readvariableop5
1savev2_watermark_hidden2_bias_read_readvariableop5
1savev2_cartoon_hidden2_kernel_read_readvariableop3
/savev2_cartoon_hidden2_bias_read_readvariableop7
3savev2_watermark_hidden3_kernel_read_readvariableop5
1savev2_watermark_hidden3_bias_read_readvariableop5
1savev2_cartoon_hidden3_kernel_read_readvariableop3
/savev2_cartoon_hidden3_bias_read_readvariableop7
3savev2_watermark_hidden4_kernel_read_readvariableop5
1savev2_watermark_hidden4_bias_read_readvariableop4
0savev2_cartoon_output_kernel_read_readvariableop2
.savev2_cartoon_output_bias_read_readvariableop6
2savev2_watermark_output_kernel_read_readvariableop4
0savev2_watermark_output_bias_read_readvariableop2
.savev2_block1_conv1_kernel_read_readvariableop0
,savev2_block1_conv1_bias_read_readvariableop2
.savev2_block1_conv2_kernel_read_readvariableop0
,savev2_block1_conv2_bias_read_readvariableop2
.savev2_block2_conv1_kernel_read_readvariableop0
,savev2_block2_conv1_bias_read_readvariableop2
.savev2_block2_conv2_kernel_read_readvariableop0
,savev2_block2_conv2_bias_read_readvariableop2
.savev2_block3_conv1_kernel_read_readvariableop0
,savev2_block3_conv1_bias_read_readvariableop2
.savev2_block3_conv2_kernel_read_readvariableop0
,savev2_block3_conv2_bias_read_readvariableop2
.savev2_block3_conv3_kernel_read_readvariableop0
,savev2_block3_conv3_bias_read_readvariableop2
.savev2_block4_conv1_kernel_read_readvariableop0
,savev2_block4_conv1_bias_read_readvariableop2
.savev2_block4_conv2_kernel_read_readvariableop0
,savev2_block4_conv2_bias_read_readvariableop2
.savev2_block4_conv3_kernel_read_readvariableop0
,savev2_block4_conv3_bias_read_readvariableop2
.savev2_block5_conv1_kernel_read_readvariableop0
,savev2_block5_conv1_bias_read_readvariableop2
.savev2_block5_conv2_kernel_read_readvariableop0
,savev2_block5_conv2_bias_read_readvariableop2
.savev2_block5_conv3_kernel_read_readvariableop0
,savev2_block5_conv3_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*?
value?B?1B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*u
valuelBj1B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:03savev2_watermark_hidden1_kernel_read_readvariableop1savev2_watermark_hidden1_bias_read_readvariableop1savev2_cartoon_hidden1_kernel_read_readvariableop/savev2_cartoon_hidden1_bias_read_readvariableop3savev2_watermark_hidden2_kernel_read_readvariableop1savev2_watermark_hidden2_bias_read_readvariableop1savev2_cartoon_hidden2_kernel_read_readvariableop/savev2_cartoon_hidden2_bias_read_readvariableop3savev2_watermark_hidden3_kernel_read_readvariableop1savev2_watermark_hidden3_bias_read_readvariableop1savev2_cartoon_hidden3_kernel_read_readvariableop/savev2_cartoon_hidden3_bias_read_readvariableop3savev2_watermark_hidden4_kernel_read_readvariableop1savev2_watermark_hidden4_bias_read_readvariableop0savev2_cartoon_output_kernel_read_readvariableop.savev2_cartoon_output_bias_read_readvariableop2savev2_watermark_output_kernel_read_readvariableop0savev2_watermark_output_bias_read_readvariableop.savev2_block1_conv1_kernel_read_readvariableop,savev2_block1_conv1_bias_read_readvariableop.savev2_block1_conv2_kernel_read_readvariableop,savev2_block1_conv2_bias_read_readvariableop.savev2_block2_conv1_kernel_read_readvariableop,savev2_block2_conv1_bias_read_readvariableop.savev2_block2_conv2_kernel_read_readvariableop,savev2_block2_conv2_bias_read_readvariableop.savev2_block3_conv1_kernel_read_readvariableop,savev2_block3_conv1_bias_read_readvariableop.savev2_block3_conv2_kernel_read_readvariableop,savev2_block3_conv2_bias_read_readvariableop.savev2_block3_conv3_kernel_read_readvariableop,savev2_block3_conv3_bias_read_readvariableop.savev2_block4_conv1_kernel_read_readvariableop,savev2_block4_conv1_bias_read_readvariableop.savev2_block4_conv2_kernel_read_readvariableop,savev2_block4_conv2_bias_read_readvariableop.savev2_block4_conv3_kernel_read_readvariableop,savev2_block4_conv3_bias_read_readvariableop.savev2_block5_conv1_kernel_read_readvariableop,savev2_block5_conv1_bias_read_readvariableop.savev2_block5_conv2_kernel_read_readvariableop,savev2_block5_conv2_bias_read_readvariableop.savev2_block5_conv3_kernel_read_readvariableop,savev2_block5_conv3_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *?
dtypes5
3212
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??@:@:
??@:@:@ : :@:: ::
:
:
:
:
::
::@:@:@@:@:@?:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??@: 

_output_shapes
:@:&"
 
_output_shapes
:
??@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

:@: 

_output_shapes
::$	 

_output_shapes

: : 


_output_shapes
::$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::$ 

_output_shapes

:
: 

_output_shapes
::,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:! 

_output_shapes	
:?:.!*
(
_output_shapes
:??:!"

_output_shapes	
:?:.#*
(
_output_shapes
:??:!$

_output_shapes	
:?:.%*
(
_output_shapes
:??:!&

_output_shapes	
:?:.'*
(
_output_shapes
:??:!(

_output_shapes	
:?:.)*
(
_output_shapes
:??:!*

_output_shapes	
:?:.+*
(
_output_shapes
:??:!,

_output_shapes	
:?:-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: 
?
?
H__inference_block2_conv1_layer_call_and_return_conditional_losses_106160

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2	
BiasAddc
ReluReluBiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?

?
K__inference_cartoon_hidden2_layer_call_and_return_conditional_losses_107070

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
H__inference_block1_conv2_layer_call_and_return_conditional_losses_109116

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
??
?-
!__inference__wrapped_model_106047
input_31T
:model_28_vgg16_block1_conv1_conv2d_readvariableop_resource:@I
;model_28_vgg16_block1_conv1_biasadd_readvariableop_resource:@T
:model_28_vgg16_block1_conv2_conv2d_readvariableop_resource:@@I
;model_28_vgg16_block1_conv2_biasadd_readvariableop_resource:@U
:model_28_vgg16_block2_conv1_conv2d_readvariableop_resource:@?J
;model_28_vgg16_block2_conv1_biasadd_readvariableop_resource:	?V
:model_28_vgg16_block2_conv2_conv2d_readvariableop_resource:??J
;model_28_vgg16_block2_conv2_biasadd_readvariableop_resource:	?V
:model_28_vgg16_block3_conv1_conv2d_readvariableop_resource:??J
;model_28_vgg16_block3_conv1_biasadd_readvariableop_resource:	?V
:model_28_vgg16_block3_conv2_conv2d_readvariableop_resource:??J
;model_28_vgg16_block3_conv2_biasadd_readvariableop_resource:	?V
:model_28_vgg16_block3_conv3_conv2d_readvariableop_resource:??J
;model_28_vgg16_block3_conv3_biasadd_readvariableop_resource:	?V
:model_28_vgg16_block4_conv1_conv2d_readvariableop_resource:??J
;model_28_vgg16_block4_conv1_biasadd_readvariableop_resource:	?V
:model_28_vgg16_block4_conv2_conv2d_readvariableop_resource:??J
;model_28_vgg16_block4_conv2_biasadd_readvariableop_resource:	?V
:model_28_vgg16_block4_conv3_conv2d_readvariableop_resource:??J
;model_28_vgg16_block4_conv3_biasadd_readvariableop_resource:	?V
:model_28_vgg16_block5_conv1_conv2d_readvariableop_resource:??J
;model_28_vgg16_block5_conv1_biasadd_readvariableop_resource:	?V
:model_28_vgg16_block5_conv2_conv2d_readvariableop_resource:??J
;model_28_vgg16_block5_conv2_biasadd_readvariableop_resource:	?V
:model_28_vgg16_block5_conv3_conv2d_readvariableop_resource:??J
;model_28_vgg16_block5_conv3_biasadd_readvariableop_resource:	?M
9model_28_watermark_hidden1_matmul_readvariableop_resource:
??@H
:model_28_watermark_hidden1_biasadd_readvariableop_resource:@K
9model_28_watermark_hidden2_matmul_readvariableop_resource:@ H
:model_28_watermark_hidden2_biasadd_readvariableop_resource: K
7model_28_cartoon_hidden1_matmul_readvariableop_resource:
??@F
8model_28_cartoon_hidden1_biasadd_readvariableop_resource:@K
9model_28_watermark_hidden3_matmul_readvariableop_resource: H
:model_28_watermark_hidden3_biasadd_readvariableop_resource:I
7model_28_cartoon_hidden2_matmul_readvariableop_resource:@F
8model_28_cartoon_hidden2_biasadd_readvariableop_resource:K
9model_28_watermark_hidden4_matmul_readvariableop_resource:
H
:model_28_watermark_hidden4_biasadd_readvariableop_resource:
I
7model_28_cartoon_hidden3_matmul_readvariableop_resource:
F
8model_28_cartoon_hidden3_biasadd_readvariableop_resource:
J
8model_28_watermark_output_matmul_readvariableop_resource:
G
9model_28_watermark_output_biasadd_readvariableop_resource:H
6model_28_cartoon_output_matmul_readvariableop_resource:
E
7model_28_cartoon_output_biasadd_readvariableop_resource:
identity

identity_1??/model_28/cartoon_hidden1/BiasAdd/ReadVariableOp?.model_28/cartoon_hidden1/MatMul/ReadVariableOp?/model_28/cartoon_hidden2/BiasAdd/ReadVariableOp?.model_28/cartoon_hidden2/MatMul/ReadVariableOp?/model_28/cartoon_hidden3/BiasAdd/ReadVariableOp?.model_28/cartoon_hidden3/MatMul/ReadVariableOp?.model_28/cartoon_output/BiasAdd/ReadVariableOp?-model_28/cartoon_output/MatMul/ReadVariableOp?2model_28/vgg16/block1_conv1/BiasAdd/ReadVariableOp?1model_28/vgg16/block1_conv1/Conv2D/ReadVariableOp?2model_28/vgg16/block1_conv2/BiasAdd/ReadVariableOp?1model_28/vgg16/block1_conv2/Conv2D/ReadVariableOp?2model_28/vgg16/block2_conv1/BiasAdd/ReadVariableOp?1model_28/vgg16/block2_conv1/Conv2D/ReadVariableOp?2model_28/vgg16/block2_conv2/BiasAdd/ReadVariableOp?1model_28/vgg16/block2_conv2/Conv2D/ReadVariableOp?2model_28/vgg16/block3_conv1/BiasAdd/ReadVariableOp?1model_28/vgg16/block3_conv1/Conv2D/ReadVariableOp?2model_28/vgg16/block3_conv2/BiasAdd/ReadVariableOp?1model_28/vgg16/block3_conv2/Conv2D/ReadVariableOp?2model_28/vgg16/block3_conv3/BiasAdd/ReadVariableOp?1model_28/vgg16/block3_conv3/Conv2D/ReadVariableOp?2model_28/vgg16/block4_conv1/BiasAdd/ReadVariableOp?1model_28/vgg16/block4_conv1/Conv2D/ReadVariableOp?2model_28/vgg16/block4_conv2/BiasAdd/ReadVariableOp?1model_28/vgg16/block4_conv2/Conv2D/ReadVariableOp?2model_28/vgg16/block4_conv3/BiasAdd/ReadVariableOp?1model_28/vgg16/block4_conv3/Conv2D/ReadVariableOp?2model_28/vgg16/block5_conv1/BiasAdd/ReadVariableOp?1model_28/vgg16/block5_conv1/Conv2D/ReadVariableOp?2model_28/vgg16/block5_conv2/BiasAdd/ReadVariableOp?1model_28/vgg16/block5_conv2/Conv2D/ReadVariableOp?2model_28/vgg16/block5_conv3/BiasAdd/ReadVariableOp?1model_28/vgg16/block5_conv3/Conv2D/ReadVariableOp?1model_28/watermark_hidden1/BiasAdd/ReadVariableOp?0model_28/watermark_hidden1/MatMul/ReadVariableOp?1model_28/watermark_hidden2/BiasAdd/ReadVariableOp?0model_28/watermark_hidden2/MatMul/ReadVariableOp?1model_28/watermark_hidden3/BiasAdd/ReadVariableOp?0model_28/watermark_hidden3/MatMul/ReadVariableOp?1model_28/watermark_hidden4/BiasAdd/ReadVariableOp?0model_28/watermark_hidden4/MatMul/ReadVariableOp?0model_28/watermark_output/BiasAdd/ReadVariableOp?/model_28/watermark_output/MatMul/ReadVariableOp?
model_28/rescaling_24/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???;2
model_28/rescaling_24/Cast/x?
model_28/rescaling_24/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
model_28/rescaling_24/Cast_1/x?
model_28/rescaling_24/mulMulinput_31%model_28/rescaling_24/Cast/x:output:0*
T0*1
_output_shapes
:???????????2
model_28/rescaling_24/mul?
model_28/rescaling_24/addAddV2model_28/rescaling_24/mul:z:0'model_28/rescaling_24/Cast_1/x:output:0*
T0*1
_output_shapes
:???????????2
model_28/rescaling_24/add?
1model_28/vgg16/block1_conv1/Conv2D/ReadVariableOpReadVariableOp:model_28_vgg16_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype023
1model_28/vgg16/block1_conv1/Conv2D/ReadVariableOp?
"model_28/vgg16/block1_conv1/Conv2DConv2Dmodel_28/rescaling_24/add:z:09model_28/vgg16/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2$
"model_28/vgg16/block1_conv1/Conv2D?
2model_28/vgg16/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp;model_28_vgg16_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype024
2model_28/vgg16/block1_conv1/BiasAdd/ReadVariableOp?
#model_28/vgg16/block1_conv1/BiasAddBiasAdd+model_28/vgg16/block1_conv1/Conv2D:output:0:model_28/vgg16/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2%
#model_28/vgg16/block1_conv1/BiasAdd?
 model_28/vgg16/block1_conv1/ReluRelu,model_28/vgg16/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2"
 model_28/vgg16/block1_conv1/Relu?
1model_28/vgg16/block1_conv2/Conv2D/ReadVariableOpReadVariableOp:model_28_vgg16_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype023
1model_28/vgg16/block1_conv2/Conv2D/ReadVariableOp?
"model_28/vgg16/block1_conv2/Conv2DConv2D.model_28/vgg16/block1_conv1/Relu:activations:09model_28/vgg16/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2$
"model_28/vgg16/block1_conv2/Conv2D?
2model_28/vgg16/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp;model_28_vgg16_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype024
2model_28/vgg16/block1_conv2/BiasAdd/ReadVariableOp?
#model_28/vgg16/block1_conv2/BiasAddBiasAdd+model_28/vgg16/block1_conv2/Conv2D:output:0:model_28/vgg16/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2%
#model_28/vgg16/block1_conv2/BiasAdd?
 model_28/vgg16/block1_conv2/ReluRelu,model_28/vgg16/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2"
 model_28/vgg16/block1_conv2/Relu?
"model_28/vgg16/block1_pool/MaxPoolMaxPool.model_28/vgg16/block1_conv2/Relu:activations:0*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
2$
"model_28/vgg16/block1_pool/MaxPool?
1model_28/vgg16/block2_conv1/Conv2D/ReadVariableOpReadVariableOp:model_28_vgg16_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype023
1model_28/vgg16/block2_conv1/Conv2D/ReadVariableOp?
"model_28/vgg16/block2_conv1/Conv2DConv2D+model_28/vgg16/block1_pool/MaxPool:output:09model_28/vgg16/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2$
"model_28/vgg16/block2_conv1/Conv2D?
2model_28/vgg16/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp;model_28_vgg16_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype024
2model_28/vgg16/block2_conv1/BiasAdd/ReadVariableOp?
#model_28/vgg16/block2_conv1/BiasAddBiasAdd+model_28/vgg16/block2_conv1/Conv2D:output:0:model_28/vgg16/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2%
#model_28/vgg16/block2_conv1/BiasAdd?
 model_28/vgg16/block2_conv1/ReluRelu,model_28/vgg16/block2_conv1/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2"
 model_28/vgg16/block2_conv1/Relu?
1model_28/vgg16/block2_conv2/Conv2D/ReadVariableOpReadVariableOp:model_28_vgg16_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype023
1model_28/vgg16/block2_conv2/Conv2D/ReadVariableOp?
"model_28/vgg16/block2_conv2/Conv2DConv2D.model_28/vgg16/block2_conv1/Relu:activations:09model_28/vgg16/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2$
"model_28/vgg16/block2_conv2/Conv2D?
2model_28/vgg16/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp;model_28_vgg16_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype024
2model_28/vgg16/block2_conv2/BiasAdd/ReadVariableOp?
#model_28/vgg16/block2_conv2/BiasAddBiasAdd+model_28/vgg16/block2_conv2/Conv2D:output:0:model_28/vgg16/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2%
#model_28/vgg16/block2_conv2/BiasAdd?
 model_28/vgg16/block2_conv2/ReluRelu,model_28/vgg16/block2_conv2/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2"
 model_28/vgg16/block2_conv2/Relu?
"model_28/vgg16/block2_pool/MaxPoolMaxPool.model_28/vgg16/block2_conv2/Relu:activations:0*0
_output_shapes
:?????????KK?*
ksize
*
paddingVALID*
strides
2$
"model_28/vgg16/block2_pool/MaxPool?
1model_28/vgg16/block3_conv1/Conv2D/ReadVariableOpReadVariableOp:model_28_vgg16_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype023
1model_28/vgg16/block3_conv1/Conv2D/ReadVariableOp?
"model_28/vgg16/block3_conv1/Conv2DConv2D+model_28/vgg16/block2_pool/MaxPool:output:09model_28/vgg16/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?*
paddingSAME*
strides
2$
"model_28/vgg16/block3_conv1/Conv2D?
2model_28/vgg16/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp;model_28_vgg16_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype024
2model_28/vgg16/block3_conv1/BiasAdd/ReadVariableOp?
#model_28/vgg16/block3_conv1/BiasAddBiasAdd+model_28/vgg16/block3_conv1/Conv2D:output:0:model_28/vgg16/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?2%
#model_28/vgg16/block3_conv1/BiasAdd?
 model_28/vgg16/block3_conv1/ReluRelu,model_28/vgg16/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????KK?2"
 model_28/vgg16/block3_conv1/Relu?
1model_28/vgg16/block3_conv2/Conv2D/ReadVariableOpReadVariableOp:model_28_vgg16_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype023
1model_28/vgg16/block3_conv2/Conv2D/ReadVariableOp?
"model_28/vgg16/block3_conv2/Conv2DConv2D.model_28/vgg16/block3_conv1/Relu:activations:09model_28/vgg16/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?*
paddingSAME*
strides
2$
"model_28/vgg16/block3_conv2/Conv2D?
2model_28/vgg16/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp;model_28_vgg16_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype024
2model_28/vgg16/block3_conv2/BiasAdd/ReadVariableOp?
#model_28/vgg16/block3_conv2/BiasAddBiasAdd+model_28/vgg16/block3_conv2/Conv2D:output:0:model_28/vgg16/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?2%
#model_28/vgg16/block3_conv2/BiasAdd?
 model_28/vgg16/block3_conv2/ReluRelu,model_28/vgg16/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????KK?2"
 model_28/vgg16/block3_conv2/Relu?
1model_28/vgg16/block3_conv3/Conv2D/ReadVariableOpReadVariableOp:model_28_vgg16_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype023
1model_28/vgg16/block3_conv3/Conv2D/ReadVariableOp?
"model_28/vgg16/block3_conv3/Conv2DConv2D.model_28/vgg16/block3_conv2/Relu:activations:09model_28/vgg16/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?*
paddingSAME*
strides
2$
"model_28/vgg16/block3_conv3/Conv2D?
2model_28/vgg16/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp;model_28_vgg16_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype024
2model_28/vgg16/block3_conv3/BiasAdd/ReadVariableOp?
#model_28/vgg16/block3_conv3/BiasAddBiasAdd+model_28/vgg16/block3_conv3/Conv2D:output:0:model_28/vgg16/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?2%
#model_28/vgg16/block3_conv3/BiasAdd?
 model_28/vgg16/block3_conv3/ReluRelu,model_28/vgg16/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????KK?2"
 model_28/vgg16/block3_conv3/Relu?
"model_28/vgg16/block3_pool/MaxPoolMaxPool.model_28/vgg16/block3_conv3/Relu:activations:0*0
_output_shapes
:?????????%%?*
ksize
*
paddingVALID*
strides
2$
"model_28/vgg16/block3_pool/MaxPool?
1model_28/vgg16/block4_conv1/Conv2D/ReadVariableOpReadVariableOp:model_28_vgg16_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype023
1model_28/vgg16/block4_conv1/Conv2D/ReadVariableOp?
"model_28/vgg16/block4_conv1/Conv2DConv2D+model_28/vgg16/block3_pool/MaxPool:output:09model_28/vgg16/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?*
paddingSAME*
strides
2$
"model_28/vgg16/block4_conv1/Conv2D?
2model_28/vgg16/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp;model_28_vgg16_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype024
2model_28/vgg16/block4_conv1/BiasAdd/ReadVariableOp?
#model_28/vgg16/block4_conv1/BiasAddBiasAdd+model_28/vgg16/block4_conv1/Conv2D:output:0:model_28/vgg16/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?2%
#model_28/vgg16/block4_conv1/BiasAdd?
 model_28/vgg16/block4_conv1/ReluRelu,model_28/vgg16/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????%%?2"
 model_28/vgg16/block4_conv1/Relu?
1model_28/vgg16/block4_conv2/Conv2D/ReadVariableOpReadVariableOp:model_28_vgg16_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype023
1model_28/vgg16/block4_conv2/Conv2D/ReadVariableOp?
"model_28/vgg16/block4_conv2/Conv2DConv2D.model_28/vgg16/block4_conv1/Relu:activations:09model_28/vgg16/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?*
paddingSAME*
strides
2$
"model_28/vgg16/block4_conv2/Conv2D?
2model_28/vgg16/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp;model_28_vgg16_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype024
2model_28/vgg16/block4_conv2/BiasAdd/ReadVariableOp?
#model_28/vgg16/block4_conv2/BiasAddBiasAdd+model_28/vgg16/block4_conv2/Conv2D:output:0:model_28/vgg16/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?2%
#model_28/vgg16/block4_conv2/BiasAdd?
 model_28/vgg16/block4_conv2/ReluRelu,model_28/vgg16/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????%%?2"
 model_28/vgg16/block4_conv2/Relu?
1model_28/vgg16/block4_conv3/Conv2D/ReadVariableOpReadVariableOp:model_28_vgg16_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype023
1model_28/vgg16/block4_conv3/Conv2D/ReadVariableOp?
"model_28/vgg16/block4_conv3/Conv2DConv2D.model_28/vgg16/block4_conv2/Relu:activations:09model_28/vgg16/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?*
paddingSAME*
strides
2$
"model_28/vgg16/block4_conv3/Conv2D?
2model_28/vgg16/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp;model_28_vgg16_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype024
2model_28/vgg16/block4_conv3/BiasAdd/ReadVariableOp?
#model_28/vgg16/block4_conv3/BiasAddBiasAdd+model_28/vgg16/block4_conv3/Conv2D:output:0:model_28/vgg16/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?2%
#model_28/vgg16/block4_conv3/BiasAdd?
 model_28/vgg16/block4_conv3/ReluRelu,model_28/vgg16/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????%%?2"
 model_28/vgg16/block4_conv3/Relu?
"model_28/vgg16/block4_pool/MaxPoolMaxPool.model_28/vgg16/block4_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2$
"model_28/vgg16/block4_pool/MaxPool?
1model_28/vgg16/block5_conv1/Conv2D/ReadVariableOpReadVariableOp:model_28_vgg16_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype023
1model_28/vgg16/block5_conv1/Conv2D/ReadVariableOp?
"model_28/vgg16/block5_conv1/Conv2DConv2D+model_28/vgg16/block4_pool/MaxPool:output:09model_28/vgg16/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2$
"model_28/vgg16/block5_conv1/Conv2D?
2model_28/vgg16/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp;model_28_vgg16_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype024
2model_28/vgg16/block5_conv1/BiasAdd/ReadVariableOp?
#model_28/vgg16/block5_conv1/BiasAddBiasAdd+model_28/vgg16/block5_conv1/Conv2D:output:0:model_28/vgg16/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2%
#model_28/vgg16/block5_conv1/BiasAdd?
 model_28/vgg16/block5_conv1/ReluRelu,model_28/vgg16/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2"
 model_28/vgg16/block5_conv1/Relu?
1model_28/vgg16/block5_conv2/Conv2D/ReadVariableOpReadVariableOp:model_28_vgg16_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype023
1model_28/vgg16/block5_conv2/Conv2D/ReadVariableOp?
"model_28/vgg16/block5_conv2/Conv2DConv2D.model_28/vgg16/block5_conv1/Relu:activations:09model_28/vgg16/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2$
"model_28/vgg16/block5_conv2/Conv2D?
2model_28/vgg16/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp;model_28_vgg16_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype024
2model_28/vgg16/block5_conv2/BiasAdd/ReadVariableOp?
#model_28/vgg16/block5_conv2/BiasAddBiasAdd+model_28/vgg16/block5_conv2/Conv2D:output:0:model_28/vgg16/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2%
#model_28/vgg16/block5_conv2/BiasAdd?
 model_28/vgg16/block5_conv2/ReluRelu,model_28/vgg16/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2"
 model_28/vgg16/block5_conv2/Relu?
1model_28/vgg16/block5_conv3/Conv2D/ReadVariableOpReadVariableOp:model_28_vgg16_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype023
1model_28/vgg16/block5_conv3/Conv2D/ReadVariableOp?
"model_28/vgg16/block5_conv3/Conv2DConv2D.model_28/vgg16/block5_conv2/Relu:activations:09model_28/vgg16/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2$
"model_28/vgg16/block5_conv3/Conv2D?
2model_28/vgg16/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp;model_28_vgg16_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype024
2model_28/vgg16/block5_conv3/BiasAdd/ReadVariableOp?
#model_28/vgg16/block5_conv3/BiasAddBiasAdd+model_28/vgg16/block5_conv3/Conv2D:output:0:model_28/vgg16/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2%
#model_28/vgg16/block5_conv3/BiasAdd?
 model_28/vgg16/block5_conv3/ReluRelu,model_28/vgg16/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2"
 model_28/vgg16/block5_conv3/Relu?
"model_28/vgg16/block5_pool/MaxPoolMaxPool.model_28/vgg16/block5_conv3/Relu:activations:0*0
_output_shapes
:?????????		?*
ksize
*
paddingVALID*
strides
2$
"model_28/vgg16/block5_pool/MaxPool?
model_28/flatten_28/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ?  2
model_28/flatten_28/Const?
model_28/flatten_28/ReshapeReshape+model_28/vgg16/block5_pool/MaxPool:output:0"model_28/flatten_28/Const:output:0*
T0*)
_output_shapes
:???????????2
model_28/flatten_28/Reshape?
0model_28/watermark_hidden1/MatMul/ReadVariableOpReadVariableOp9model_28_watermark_hidden1_matmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype022
0model_28/watermark_hidden1/MatMul/ReadVariableOp?
!model_28/watermark_hidden1/MatMulMatMul$model_28/flatten_28/Reshape:output:08model_28/watermark_hidden1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2#
!model_28/watermark_hidden1/MatMul?
1model_28/watermark_hidden1/BiasAdd/ReadVariableOpReadVariableOp:model_28_watermark_hidden1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype023
1model_28/watermark_hidden1/BiasAdd/ReadVariableOp?
"model_28/watermark_hidden1/BiasAddBiasAdd+model_28/watermark_hidden1/MatMul:product:09model_28/watermark_hidden1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2$
"model_28/watermark_hidden1/BiasAdd?
model_28/watermark_hidden1/ReluRelu+model_28/watermark_hidden1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2!
model_28/watermark_hidden1/Relu?
0model_28/watermark_hidden2/MatMul/ReadVariableOpReadVariableOp9model_28_watermark_hidden2_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype022
0model_28/watermark_hidden2/MatMul/ReadVariableOp?
!model_28/watermark_hidden2/MatMulMatMul-model_28/watermark_hidden1/Relu:activations:08model_28/watermark_hidden2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2#
!model_28/watermark_hidden2/MatMul?
1model_28/watermark_hidden2/BiasAdd/ReadVariableOpReadVariableOp:model_28_watermark_hidden2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype023
1model_28/watermark_hidden2/BiasAdd/ReadVariableOp?
"model_28/watermark_hidden2/BiasAddBiasAdd+model_28/watermark_hidden2/MatMul:product:09model_28/watermark_hidden2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2$
"model_28/watermark_hidden2/BiasAdd?
model_28/watermark_hidden2/ReluRelu+model_28/watermark_hidden2/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2!
model_28/watermark_hidden2/Relu?
.model_28/cartoon_hidden1/MatMul/ReadVariableOpReadVariableOp7model_28_cartoon_hidden1_matmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype020
.model_28/cartoon_hidden1/MatMul/ReadVariableOp?
model_28/cartoon_hidden1/MatMulMatMul$model_28/flatten_28/Reshape:output:06model_28/cartoon_hidden1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2!
model_28/cartoon_hidden1/MatMul?
/model_28/cartoon_hidden1/BiasAdd/ReadVariableOpReadVariableOp8model_28_cartoon_hidden1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/model_28/cartoon_hidden1/BiasAdd/ReadVariableOp?
 model_28/cartoon_hidden1/BiasAddBiasAdd)model_28/cartoon_hidden1/MatMul:product:07model_28/cartoon_hidden1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2"
 model_28/cartoon_hidden1/BiasAdd?
model_28/cartoon_hidden1/ReluRelu)model_28/cartoon_hidden1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model_28/cartoon_hidden1/Relu?
0model_28/watermark_hidden3/MatMul/ReadVariableOpReadVariableOp9model_28_watermark_hidden3_matmul_readvariableop_resource*
_output_shapes

: *
dtype022
0model_28/watermark_hidden3/MatMul/ReadVariableOp?
!model_28/watermark_hidden3/MatMulMatMul-model_28/watermark_hidden2/Relu:activations:08model_28/watermark_hidden3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!model_28/watermark_hidden3/MatMul?
1model_28/watermark_hidden3/BiasAdd/ReadVariableOpReadVariableOp:model_28_watermark_hidden3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1model_28/watermark_hidden3/BiasAdd/ReadVariableOp?
"model_28/watermark_hidden3/BiasAddBiasAdd+model_28/watermark_hidden3/MatMul:product:09model_28/watermark_hidden3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2$
"model_28/watermark_hidden3/BiasAdd?
model_28/watermark_hidden3/ReluRelu+model_28/watermark_hidden3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2!
model_28/watermark_hidden3/Relu?
.model_28/cartoon_hidden2/MatMul/ReadVariableOpReadVariableOp7model_28_cartoon_hidden2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype020
.model_28/cartoon_hidden2/MatMul/ReadVariableOp?
model_28/cartoon_hidden2/MatMulMatMul+model_28/cartoon_hidden1/Relu:activations:06model_28/cartoon_hidden2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
model_28/cartoon_hidden2/MatMul?
/model_28/cartoon_hidden2/BiasAdd/ReadVariableOpReadVariableOp8model_28_cartoon_hidden2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/model_28/cartoon_hidden2/BiasAdd/ReadVariableOp?
 model_28/cartoon_hidden2/BiasAddBiasAdd)model_28/cartoon_hidden2/MatMul:product:07model_28/cartoon_hidden2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 model_28/cartoon_hidden2/BiasAdd?
model_28/cartoon_hidden2/ReluRelu)model_28/cartoon_hidden2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_28/cartoon_hidden2/Relu?
0model_28/watermark_hidden4/MatMul/ReadVariableOpReadVariableOp9model_28_watermark_hidden4_matmul_readvariableop_resource*
_output_shapes

:
*
dtype022
0model_28/watermark_hidden4/MatMul/ReadVariableOp?
!model_28/watermark_hidden4/MatMulMatMul-model_28/watermark_hidden3/Relu:activations:08model_28/watermark_hidden4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2#
!model_28/watermark_hidden4/MatMul?
1model_28/watermark_hidden4/BiasAdd/ReadVariableOpReadVariableOp:model_28_watermark_hidden4_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype023
1model_28/watermark_hidden4/BiasAdd/ReadVariableOp?
"model_28/watermark_hidden4/BiasAddBiasAdd+model_28/watermark_hidden4/MatMul:product:09model_28/watermark_hidden4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2$
"model_28/watermark_hidden4/BiasAdd?
model_28/watermark_hidden4/ReluRelu+model_28/watermark_hidden4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2!
model_28/watermark_hidden4/Relu?
.model_28/cartoon_hidden3/MatMul/ReadVariableOpReadVariableOp7model_28_cartoon_hidden3_matmul_readvariableop_resource*
_output_shapes

:
*
dtype020
.model_28/cartoon_hidden3/MatMul/ReadVariableOp?
model_28/cartoon_hidden3/MatMulMatMul+model_28/cartoon_hidden2/Relu:activations:06model_28/cartoon_hidden3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2!
model_28/cartoon_hidden3/MatMul?
/model_28/cartoon_hidden3/BiasAdd/ReadVariableOpReadVariableOp8model_28_cartoon_hidden3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype021
/model_28/cartoon_hidden3/BiasAdd/ReadVariableOp?
 model_28/cartoon_hidden3/BiasAddBiasAdd)model_28/cartoon_hidden3/MatMul:product:07model_28/cartoon_hidden3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2"
 model_28/cartoon_hidden3/BiasAdd?
model_28/cartoon_hidden3/ReluRelu)model_28/cartoon_hidden3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
model_28/cartoon_hidden3/Relu?
/model_28/watermark_output/MatMul/ReadVariableOpReadVariableOp8model_28_watermark_output_matmul_readvariableop_resource*
_output_shapes

:
*
dtype021
/model_28/watermark_output/MatMul/ReadVariableOp?
 model_28/watermark_output/MatMulMatMul-model_28/watermark_hidden4/Relu:activations:07model_28/watermark_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 model_28/watermark_output/MatMul?
0model_28/watermark_output/BiasAdd/ReadVariableOpReadVariableOp9model_28_watermark_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0model_28/watermark_output/BiasAdd/ReadVariableOp?
!model_28/watermark_output/BiasAddBiasAdd*model_28/watermark_output/MatMul:product:08model_28/watermark_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!model_28/watermark_output/BiasAdd?
!model_28/watermark_output/SigmoidSigmoid*model_28/watermark_output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2#
!model_28/watermark_output/Sigmoid?
-model_28/cartoon_output/MatMul/ReadVariableOpReadVariableOp6model_28_cartoon_output_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02/
-model_28/cartoon_output/MatMul/ReadVariableOp?
model_28/cartoon_output/MatMulMatMul+model_28/cartoon_hidden3/Relu:activations:05model_28/cartoon_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
model_28/cartoon_output/MatMul?
.model_28/cartoon_output/BiasAdd/ReadVariableOpReadVariableOp7model_28_cartoon_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.model_28/cartoon_output/BiasAdd/ReadVariableOp?
model_28/cartoon_output/BiasAddBiasAdd(model_28/cartoon_output/MatMul:product:06model_28/cartoon_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
model_28/cartoon_output/BiasAdd?
model_28/cartoon_output/SigmoidSigmoid(model_28/cartoon_output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2!
model_28/cartoon_output/Sigmoid?
IdentityIdentity#model_28/cartoon_output/Sigmoid:y:00^model_28/cartoon_hidden1/BiasAdd/ReadVariableOp/^model_28/cartoon_hidden1/MatMul/ReadVariableOp0^model_28/cartoon_hidden2/BiasAdd/ReadVariableOp/^model_28/cartoon_hidden2/MatMul/ReadVariableOp0^model_28/cartoon_hidden3/BiasAdd/ReadVariableOp/^model_28/cartoon_hidden3/MatMul/ReadVariableOp/^model_28/cartoon_output/BiasAdd/ReadVariableOp.^model_28/cartoon_output/MatMul/ReadVariableOp3^model_28/vgg16/block1_conv1/BiasAdd/ReadVariableOp2^model_28/vgg16/block1_conv1/Conv2D/ReadVariableOp3^model_28/vgg16/block1_conv2/BiasAdd/ReadVariableOp2^model_28/vgg16/block1_conv2/Conv2D/ReadVariableOp3^model_28/vgg16/block2_conv1/BiasAdd/ReadVariableOp2^model_28/vgg16/block2_conv1/Conv2D/ReadVariableOp3^model_28/vgg16/block2_conv2/BiasAdd/ReadVariableOp2^model_28/vgg16/block2_conv2/Conv2D/ReadVariableOp3^model_28/vgg16/block3_conv1/BiasAdd/ReadVariableOp2^model_28/vgg16/block3_conv1/Conv2D/ReadVariableOp3^model_28/vgg16/block3_conv2/BiasAdd/ReadVariableOp2^model_28/vgg16/block3_conv2/Conv2D/ReadVariableOp3^model_28/vgg16/block3_conv3/BiasAdd/ReadVariableOp2^model_28/vgg16/block3_conv3/Conv2D/ReadVariableOp3^model_28/vgg16/block4_conv1/BiasAdd/ReadVariableOp2^model_28/vgg16/block4_conv1/Conv2D/ReadVariableOp3^model_28/vgg16/block4_conv2/BiasAdd/ReadVariableOp2^model_28/vgg16/block4_conv2/Conv2D/ReadVariableOp3^model_28/vgg16/block4_conv3/BiasAdd/ReadVariableOp2^model_28/vgg16/block4_conv3/Conv2D/ReadVariableOp3^model_28/vgg16/block5_conv1/BiasAdd/ReadVariableOp2^model_28/vgg16/block5_conv1/Conv2D/ReadVariableOp3^model_28/vgg16/block5_conv2/BiasAdd/ReadVariableOp2^model_28/vgg16/block5_conv2/Conv2D/ReadVariableOp3^model_28/vgg16/block5_conv3/BiasAdd/ReadVariableOp2^model_28/vgg16/block5_conv3/Conv2D/ReadVariableOp2^model_28/watermark_hidden1/BiasAdd/ReadVariableOp1^model_28/watermark_hidden1/MatMul/ReadVariableOp2^model_28/watermark_hidden2/BiasAdd/ReadVariableOp1^model_28/watermark_hidden2/MatMul/ReadVariableOp2^model_28/watermark_hidden3/BiasAdd/ReadVariableOp1^model_28/watermark_hidden3/MatMul/ReadVariableOp2^model_28/watermark_hidden4/BiasAdd/ReadVariableOp1^model_28/watermark_hidden4/MatMul/ReadVariableOp1^model_28/watermark_output/BiasAdd/ReadVariableOp0^model_28/watermark_output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity%model_28/watermark_output/Sigmoid:y:00^model_28/cartoon_hidden1/BiasAdd/ReadVariableOp/^model_28/cartoon_hidden1/MatMul/ReadVariableOp0^model_28/cartoon_hidden2/BiasAdd/ReadVariableOp/^model_28/cartoon_hidden2/MatMul/ReadVariableOp0^model_28/cartoon_hidden3/BiasAdd/ReadVariableOp/^model_28/cartoon_hidden3/MatMul/ReadVariableOp/^model_28/cartoon_output/BiasAdd/ReadVariableOp.^model_28/cartoon_output/MatMul/ReadVariableOp3^model_28/vgg16/block1_conv1/BiasAdd/ReadVariableOp2^model_28/vgg16/block1_conv1/Conv2D/ReadVariableOp3^model_28/vgg16/block1_conv2/BiasAdd/ReadVariableOp2^model_28/vgg16/block1_conv2/Conv2D/ReadVariableOp3^model_28/vgg16/block2_conv1/BiasAdd/ReadVariableOp2^model_28/vgg16/block2_conv1/Conv2D/ReadVariableOp3^model_28/vgg16/block2_conv2/BiasAdd/ReadVariableOp2^model_28/vgg16/block2_conv2/Conv2D/ReadVariableOp3^model_28/vgg16/block3_conv1/BiasAdd/ReadVariableOp2^model_28/vgg16/block3_conv1/Conv2D/ReadVariableOp3^model_28/vgg16/block3_conv2/BiasAdd/ReadVariableOp2^model_28/vgg16/block3_conv2/Conv2D/ReadVariableOp3^model_28/vgg16/block3_conv3/BiasAdd/ReadVariableOp2^model_28/vgg16/block3_conv3/Conv2D/ReadVariableOp3^model_28/vgg16/block4_conv1/BiasAdd/ReadVariableOp2^model_28/vgg16/block4_conv1/Conv2D/ReadVariableOp3^model_28/vgg16/block4_conv2/BiasAdd/ReadVariableOp2^model_28/vgg16/block4_conv2/Conv2D/ReadVariableOp3^model_28/vgg16/block4_conv3/BiasAdd/ReadVariableOp2^model_28/vgg16/block4_conv3/Conv2D/ReadVariableOp3^model_28/vgg16/block5_conv1/BiasAdd/ReadVariableOp2^model_28/vgg16/block5_conv1/Conv2D/ReadVariableOp3^model_28/vgg16/block5_conv2/BiasAdd/ReadVariableOp2^model_28/vgg16/block5_conv2/Conv2D/ReadVariableOp3^model_28/vgg16/block5_conv3/BiasAdd/ReadVariableOp2^model_28/vgg16/block5_conv3/Conv2D/ReadVariableOp2^model_28/watermark_hidden1/BiasAdd/ReadVariableOp1^model_28/watermark_hidden1/MatMul/ReadVariableOp2^model_28/watermark_hidden2/BiasAdd/ReadVariableOp1^model_28/watermark_hidden2/MatMul/ReadVariableOp2^model_28/watermark_hidden3/BiasAdd/ReadVariableOp1^model_28/watermark_hidden3/MatMul/ReadVariableOp2^model_28/watermark_hidden4/BiasAdd/ReadVariableOp1^model_28/watermark_hidden4/MatMul/ReadVariableOp1^model_28/watermark_output/BiasAdd/ReadVariableOp0^model_28/watermark_output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesw
u:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/model_28/cartoon_hidden1/BiasAdd/ReadVariableOp/model_28/cartoon_hidden1/BiasAdd/ReadVariableOp2`
.model_28/cartoon_hidden1/MatMul/ReadVariableOp.model_28/cartoon_hidden1/MatMul/ReadVariableOp2b
/model_28/cartoon_hidden2/BiasAdd/ReadVariableOp/model_28/cartoon_hidden2/BiasAdd/ReadVariableOp2`
.model_28/cartoon_hidden2/MatMul/ReadVariableOp.model_28/cartoon_hidden2/MatMul/ReadVariableOp2b
/model_28/cartoon_hidden3/BiasAdd/ReadVariableOp/model_28/cartoon_hidden3/BiasAdd/ReadVariableOp2`
.model_28/cartoon_hidden3/MatMul/ReadVariableOp.model_28/cartoon_hidden3/MatMul/ReadVariableOp2`
.model_28/cartoon_output/BiasAdd/ReadVariableOp.model_28/cartoon_output/BiasAdd/ReadVariableOp2^
-model_28/cartoon_output/MatMul/ReadVariableOp-model_28/cartoon_output/MatMul/ReadVariableOp2h
2model_28/vgg16/block1_conv1/BiasAdd/ReadVariableOp2model_28/vgg16/block1_conv1/BiasAdd/ReadVariableOp2f
1model_28/vgg16/block1_conv1/Conv2D/ReadVariableOp1model_28/vgg16/block1_conv1/Conv2D/ReadVariableOp2h
2model_28/vgg16/block1_conv2/BiasAdd/ReadVariableOp2model_28/vgg16/block1_conv2/BiasAdd/ReadVariableOp2f
1model_28/vgg16/block1_conv2/Conv2D/ReadVariableOp1model_28/vgg16/block1_conv2/Conv2D/ReadVariableOp2h
2model_28/vgg16/block2_conv1/BiasAdd/ReadVariableOp2model_28/vgg16/block2_conv1/BiasAdd/ReadVariableOp2f
1model_28/vgg16/block2_conv1/Conv2D/ReadVariableOp1model_28/vgg16/block2_conv1/Conv2D/ReadVariableOp2h
2model_28/vgg16/block2_conv2/BiasAdd/ReadVariableOp2model_28/vgg16/block2_conv2/BiasAdd/ReadVariableOp2f
1model_28/vgg16/block2_conv2/Conv2D/ReadVariableOp1model_28/vgg16/block2_conv2/Conv2D/ReadVariableOp2h
2model_28/vgg16/block3_conv1/BiasAdd/ReadVariableOp2model_28/vgg16/block3_conv1/BiasAdd/ReadVariableOp2f
1model_28/vgg16/block3_conv1/Conv2D/ReadVariableOp1model_28/vgg16/block3_conv1/Conv2D/ReadVariableOp2h
2model_28/vgg16/block3_conv2/BiasAdd/ReadVariableOp2model_28/vgg16/block3_conv2/BiasAdd/ReadVariableOp2f
1model_28/vgg16/block3_conv2/Conv2D/ReadVariableOp1model_28/vgg16/block3_conv2/Conv2D/ReadVariableOp2h
2model_28/vgg16/block3_conv3/BiasAdd/ReadVariableOp2model_28/vgg16/block3_conv3/BiasAdd/ReadVariableOp2f
1model_28/vgg16/block3_conv3/Conv2D/ReadVariableOp1model_28/vgg16/block3_conv3/Conv2D/ReadVariableOp2h
2model_28/vgg16/block4_conv1/BiasAdd/ReadVariableOp2model_28/vgg16/block4_conv1/BiasAdd/ReadVariableOp2f
1model_28/vgg16/block4_conv1/Conv2D/ReadVariableOp1model_28/vgg16/block4_conv1/Conv2D/ReadVariableOp2h
2model_28/vgg16/block4_conv2/BiasAdd/ReadVariableOp2model_28/vgg16/block4_conv2/BiasAdd/ReadVariableOp2f
1model_28/vgg16/block4_conv2/Conv2D/ReadVariableOp1model_28/vgg16/block4_conv2/Conv2D/ReadVariableOp2h
2model_28/vgg16/block4_conv3/BiasAdd/ReadVariableOp2model_28/vgg16/block4_conv3/BiasAdd/ReadVariableOp2f
1model_28/vgg16/block4_conv3/Conv2D/ReadVariableOp1model_28/vgg16/block4_conv3/Conv2D/ReadVariableOp2h
2model_28/vgg16/block5_conv1/BiasAdd/ReadVariableOp2model_28/vgg16/block5_conv1/BiasAdd/ReadVariableOp2f
1model_28/vgg16/block5_conv1/Conv2D/ReadVariableOp1model_28/vgg16/block5_conv1/Conv2D/ReadVariableOp2h
2model_28/vgg16/block5_conv2/BiasAdd/ReadVariableOp2model_28/vgg16/block5_conv2/BiasAdd/ReadVariableOp2f
1model_28/vgg16/block5_conv2/Conv2D/ReadVariableOp1model_28/vgg16/block5_conv2/Conv2D/ReadVariableOp2h
2model_28/vgg16/block5_conv3/BiasAdd/ReadVariableOp2model_28/vgg16/block5_conv3/BiasAdd/ReadVariableOp2f
1model_28/vgg16/block5_conv3/Conv2D/ReadVariableOp1model_28/vgg16/block5_conv3/Conv2D/ReadVariableOp2f
1model_28/watermark_hidden1/BiasAdd/ReadVariableOp1model_28/watermark_hidden1/BiasAdd/ReadVariableOp2d
0model_28/watermark_hidden1/MatMul/ReadVariableOp0model_28/watermark_hidden1/MatMul/ReadVariableOp2f
1model_28/watermark_hidden2/BiasAdd/ReadVariableOp1model_28/watermark_hidden2/BiasAdd/ReadVariableOp2d
0model_28/watermark_hidden2/MatMul/ReadVariableOp0model_28/watermark_hidden2/MatMul/ReadVariableOp2f
1model_28/watermark_hidden3/BiasAdd/ReadVariableOp1model_28/watermark_hidden3/BiasAdd/ReadVariableOp2d
0model_28/watermark_hidden3/MatMul/ReadVariableOp0model_28/watermark_hidden3/MatMul/ReadVariableOp2f
1model_28/watermark_hidden4/BiasAdd/ReadVariableOp1model_28/watermark_hidden4/BiasAdd/ReadVariableOp2d
0model_28/watermark_hidden4/MatMul/ReadVariableOp0model_28/watermark_hidden4/MatMul/ReadVariableOp2d
0model_28/watermark_output/BiasAdd/ReadVariableOp0model_28/watermark_output/BiasAdd/ReadVariableOp2b
/model_28/watermark_output/MatMul/ReadVariableOp/model_28/watermark_output/MatMul/ReadVariableOp:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_31
?
?
2__inference_watermark_hidden1_layer_call_fn_108905

inputs
unknown:
??@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_watermark_hidden1_layer_call_and_return_conditional_losses_1070022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
M__inference_watermark_hidden3_layer_call_and_return_conditional_losses_108996

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
H__inference_block4_conv1_layer_call_and_return_conditional_losses_109236

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????%%?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????%%?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????%%?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????%%?
 
_user_specified_nameinputs
?
?
-__inference_block5_conv3_layer_call_fn_109325

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block5_conv3_layer_call_and_return_conditional_losses_1063332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?R
?
D__inference_model_28_layer_call_and_return_conditional_losses_107836
input_31&
vgg16_107735:@
vgg16_107737:@&
vgg16_107739:@@
vgg16_107741:@'
vgg16_107743:@?
vgg16_107745:	?(
vgg16_107747:??
vgg16_107749:	?(
vgg16_107751:??
vgg16_107753:	?(
vgg16_107755:??
vgg16_107757:	?(
vgg16_107759:??
vgg16_107761:	?(
vgg16_107763:??
vgg16_107765:	?(
vgg16_107767:??
vgg16_107769:	?(
vgg16_107771:??
vgg16_107773:	?(
vgg16_107775:??
vgg16_107777:	?(
vgg16_107779:??
vgg16_107781:	?(
vgg16_107783:??
vgg16_107785:	?,
watermark_hidden1_107789:
??@&
watermark_hidden1_107791:@*
watermark_hidden2_107794:@ &
watermark_hidden2_107796: *
cartoon_hidden1_107799:
??@$
cartoon_hidden1_107801:@*
watermark_hidden3_107804: &
watermark_hidden3_107806:(
cartoon_hidden2_107809:@$
cartoon_hidden2_107811:*
watermark_hidden4_107814:
&
watermark_hidden4_107816:
(
cartoon_hidden3_107819:
$
cartoon_hidden3_107821:
)
watermark_output_107824:
%
watermark_output_107826:'
cartoon_output_107829:
#
cartoon_output_107831:
identity

identity_1??'cartoon_hidden1/StatefulPartitionedCall?'cartoon_hidden2/StatefulPartitionedCall?'cartoon_hidden3/StatefulPartitionedCall?&cartoon_output/StatefulPartitionedCall?vgg16/StatefulPartitionedCall?)watermark_hidden1/StatefulPartitionedCall?)watermark_hidden2/StatefulPartitionedCall?)watermark_hidden3/StatefulPartitionedCall?)watermark_hidden4/StatefulPartitionedCall?(watermark_output/StatefulPartitionedCallo
rescaling_24/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???;2
rescaling_24/Cast/xs
rescaling_24/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling_24/Cast_1/x?
rescaling_24/mulMulinput_31rescaling_24/Cast/x:output:0*
T0*1
_output_shapes
:???????????2
rescaling_24/mul?
rescaling_24/addAddV2rescaling_24/mul:z:0rescaling_24/Cast_1/x:output:0*
T0*1
_output_shapes
:???????????2
rescaling_24/add?
vgg16/StatefulPartitionedCallStatefulPartitionedCallrescaling_24/add:z:0vgg16_107735vgg16_107737vgg16_107739vgg16_107741vgg16_107743vgg16_107745vgg16_107747vgg16_107749vgg16_107751vgg16_107753vgg16_107755vgg16_107757vgg16_107759vgg16_107761vgg16_107763vgg16_107765vgg16_107767vgg16_107769vgg16_107771vgg16_107773vgg16_107775vgg16_107777vgg16_107779vgg16_107781vgg16_107783vgg16_107785*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_vgg16_layer_call_and_return_conditional_losses_1063412
vgg16/StatefulPartitionedCall?
flatten_28/PartitionedCallPartitionedCall&vgg16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_28_layer_call_and_return_conditional_losses_1069892
flatten_28/PartitionedCall?
)watermark_hidden1/StatefulPartitionedCallStatefulPartitionedCall#flatten_28/PartitionedCall:output:0watermark_hidden1_107789watermark_hidden1_107791*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_watermark_hidden1_layer_call_and_return_conditional_losses_1070022+
)watermark_hidden1/StatefulPartitionedCall?
)watermark_hidden2/StatefulPartitionedCallStatefulPartitionedCall2watermark_hidden1/StatefulPartitionedCall:output:0watermark_hidden2_107794watermark_hidden2_107796*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_watermark_hidden2_layer_call_and_return_conditional_losses_1070192+
)watermark_hidden2/StatefulPartitionedCall?
'cartoon_hidden1/StatefulPartitionedCallStatefulPartitionedCall#flatten_28/PartitionedCall:output:0cartoon_hidden1_107799cartoon_hidden1_107801*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_cartoon_hidden1_layer_call_and_return_conditional_losses_1070362)
'cartoon_hidden1/StatefulPartitionedCall?
)watermark_hidden3/StatefulPartitionedCallStatefulPartitionedCall2watermark_hidden2/StatefulPartitionedCall:output:0watermark_hidden3_107804watermark_hidden3_107806*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_watermark_hidden3_layer_call_and_return_conditional_losses_1070532+
)watermark_hidden3/StatefulPartitionedCall?
'cartoon_hidden2/StatefulPartitionedCallStatefulPartitionedCall0cartoon_hidden1/StatefulPartitionedCall:output:0cartoon_hidden2_107809cartoon_hidden2_107811*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_cartoon_hidden2_layer_call_and_return_conditional_losses_1070702)
'cartoon_hidden2/StatefulPartitionedCall?
)watermark_hidden4/StatefulPartitionedCallStatefulPartitionedCall2watermark_hidden3/StatefulPartitionedCall:output:0watermark_hidden4_107814watermark_hidden4_107816*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_watermark_hidden4_layer_call_and_return_conditional_losses_1070872+
)watermark_hidden4/StatefulPartitionedCall?
'cartoon_hidden3/StatefulPartitionedCallStatefulPartitionedCall0cartoon_hidden2/StatefulPartitionedCall:output:0cartoon_hidden3_107819cartoon_hidden3_107821*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_cartoon_hidden3_layer_call_and_return_conditional_losses_1071042)
'cartoon_hidden3/StatefulPartitionedCall?
(watermark_output/StatefulPartitionedCallStatefulPartitionedCall2watermark_hidden4/StatefulPartitionedCall:output:0watermark_output_107824watermark_output_107826*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_watermark_output_layer_call_and_return_conditional_losses_1071212*
(watermark_output/StatefulPartitionedCall?
&cartoon_output/StatefulPartitionedCallStatefulPartitionedCall0cartoon_hidden3/StatefulPartitionedCall:output:0cartoon_output_107829cartoon_output_107831*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_cartoon_output_layer_call_and_return_conditional_losses_1071382(
&cartoon_output/StatefulPartitionedCall?
IdentityIdentity/cartoon_output/StatefulPartitionedCall:output:0(^cartoon_hidden1/StatefulPartitionedCall(^cartoon_hidden2/StatefulPartitionedCall(^cartoon_hidden3/StatefulPartitionedCall'^cartoon_output/StatefulPartitionedCall^vgg16/StatefulPartitionedCall*^watermark_hidden1/StatefulPartitionedCall*^watermark_hidden2/StatefulPartitionedCall*^watermark_hidden3/StatefulPartitionedCall*^watermark_hidden4/StatefulPartitionedCall)^watermark_output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity1watermark_output/StatefulPartitionedCall:output:0(^cartoon_hidden1/StatefulPartitionedCall(^cartoon_hidden2/StatefulPartitionedCall(^cartoon_hidden3/StatefulPartitionedCall'^cartoon_output/StatefulPartitionedCall^vgg16/StatefulPartitionedCall*^watermark_hidden1/StatefulPartitionedCall*^watermark_hidden2/StatefulPartitionedCall*^watermark_hidden3/StatefulPartitionedCall*^watermark_hidden4/StatefulPartitionedCall)^watermark_output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesw
u:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2R
'cartoon_hidden1/StatefulPartitionedCall'cartoon_hidden1/StatefulPartitionedCall2R
'cartoon_hidden2/StatefulPartitionedCall'cartoon_hidden2/StatefulPartitionedCall2R
'cartoon_hidden3/StatefulPartitionedCall'cartoon_hidden3/StatefulPartitionedCall2P
&cartoon_output/StatefulPartitionedCall&cartoon_output/StatefulPartitionedCall2>
vgg16/StatefulPartitionedCallvgg16/StatefulPartitionedCall2V
)watermark_hidden1/StatefulPartitionedCall)watermark_hidden1/StatefulPartitionedCall2V
)watermark_hidden2/StatefulPartitionedCall)watermark_hidden2/StatefulPartitionedCall2V
)watermark_hidden3/StatefulPartitionedCall)watermark_hidden3/StatefulPartitionedCall2V
)watermark_hidden4/StatefulPartitionedCall)watermark_hidden4/StatefulPartitionedCall2T
(watermark_output/StatefulPartitionedCall(watermark_output/StatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_31
?
?
H__inference_block4_conv3_layer_call_and_return_conditional_losses_106281

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????%%?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????%%?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????%%?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????%%?
 
_user_specified_nameinputs
?

?
M__inference_watermark_hidden1_layer_call_and_return_conditional_losses_107002

inputs2
matmul_readvariableop_resource:
??@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
H__inference_block1_conv2_layer_call_and_return_conditional_losses_106142

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
)__inference_model_28_layer_call_fn_107239
input_31!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?

unknown_25:
??@

unknown_26:@

unknown_27:@ 

unknown_28: 

unknown_29:
??@

unknown_30:@

unknown_31: 

unknown_32:

unknown_33:@

unknown_34:

unknown_35:


unknown_36:


unknown_37:


unknown_38:


unknown_39:


unknown_40:

unknown_41:


unknown_42:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_31unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_28_layer_call_and_return_conditional_losses_1071462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesw
u:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_31
?
?
/__inference_cartoon_output_layer_call_fn_109045

inputs
unknown:

	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_cartoon_output_layer_call_and_return_conditional_losses_1071382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
ؖ
?
A__inference_vgg16_layer_call_and_return_conditional_losses_108785

inputsE
+block1_conv1_conv2d_readvariableop_resource:@:
,block1_conv1_biasadd_readvariableop_resource:@E
+block1_conv2_conv2d_readvariableop_resource:@@:
,block1_conv2_biasadd_readvariableop_resource:@F
+block2_conv1_conv2d_readvariableop_resource:@?;
,block2_conv1_biasadd_readvariableop_resource:	?G
+block2_conv2_conv2d_readvariableop_resource:??;
,block2_conv2_biasadd_readvariableop_resource:	?G
+block3_conv1_conv2d_readvariableop_resource:??;
,block3_conv1_biasadd_readvariableop_resource:	?G
+block3_conv2_conv2d_readvariableop_resource:??;
,block3_conv2_biasadd_readvariableop_resource:	?G
+block3_conv3_conv2d_readvariableop_resource:??;
,block3_conv3_biasadd_readvariableop_resource:	?G
+block4_conv1_conv2d_readvariableop_resource:??;
,block4_conv1_biasadd_readvariableop_resource:	?G
+block4_conv2_conv2d_readvariableop_resource:??;
,block4_conv2_biasadd_readvariableop_resource:	?G
+block4_conv3_conv2d_readvariableop_resource:??;
,block4_conv3_biasadd_readvariableop_resource:	?G
+block5_conv1_conv2d_readvariableop_resource:??;
,block5_conv1_biasadd_readvariableop_resource:	?G
+block5_conv2_conv2d_readvariableop_resource:??;
,block5_conv2_biasadd_readvariableop_resource:	?G
+block5_conv3_conv2d_readvariableop_resource:??;
,block5_conv3_biasadd_readvariableop_resource:	?
identity??#block1_conv1/BiasAdd/ReadVariableOp?"block1_conv1/Conv2D/ReadVariableOp?#block1_conv2/BiasAdd/ReadVariableOp?"block1_conv2/Conv2D/ReadVariableOp?#block2_conv1/BiasAdd/ReadVariableOp?"block2_conv1/Conv2D/ReadVariableOp?#block2_conv2/BiasAdd/ReadVariableOp?"block2_conv2/Conv2D/ReadVariableOp?#block3_conv1/BiasAdd/ReadVariableOp?"block3_conv1/Conv2D/ReadVariableOp?#block3_conv2/BiasAdd/ReadVariableOp?"block3_conv2/Conv2D/ReadVariableOp?#block3_conv3/BiasAdd/ReadVariableOp?"block3_conv3/Conv2D/ReadVariableOp?#block4_conv1/BiasAdd/ReadVariableOp?"block4_conv1/Conv2D/ReadVariableOp?#block4_conv2/BiasAdd/ReadVariableOp?"block4_conv2/Conv2D/ReadVariableOp?#block4_conv3/BiasAdd/ReadVariableOp?"block4_conv3/Conv2D/ReadVariableOp?#block5_conv1/BiasAdd/ReadVariableOp?"block5_conv1/Conv2D/ReadVariableOp?#block5_conv2/BiasAdd/ReadVariableOp?"block5_conv2/Conv2D/ReadVariableOp?#block5_conv3/BiasAdd/ReadVariableOp?"block5_conv3/Conv2D/ReadVariableOp?
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02$
"block1_conv1/Conv2D/ReadVariableOp?
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
block1_conv1/Conv2D?
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv1/BiasAdd/ReadVariableOp?
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
block1_conv1/BiasAdd?
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
block1_conv1/Relu?
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02$
"block1_conv2/Conv2D/ReadVariableOp?
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
block1_conv2/Conv2D?
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv2/BiasAdd/ReadVariableOp?
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
block1_conv2/BiasAdd?
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
block1_conv2/Relu?
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
2
block1_pool/MaxPool?
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02$
"block2_conv1/Conv2D/ReadVariableOp?
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
block2_conv1/Conv2D?
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block2_conv1/BiasAdd/ReadVariableOp?
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
block2_conv1/BiasAdd?
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
block2_conv1/Relu?
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block2_conv2/Conv2D/ReadVariableOp?
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
block2_conv2/Conv2D?
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block2_conv2/BiasAdd/ReadVariableOp?
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
block2_conv2/BiasAdd?
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
block2_conv2/Relu?
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:?????????KK?*
ksize
*
paddingVALID*
strides
2
block2_pool/MaxPool?
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block3_conv1/Conv2D/ReadVariableOp?
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?*
paddingSAME*
strides
2
block3_conv1/Conv2D?
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block3_conv1/BiasAdd/ReadVariableOp?
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?2
block3_conv1/BiasAdd?
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????KK?2
block3_conv1/Relu?
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block3_conv2/Conv2D/ReadVariableOp?
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?*
paddingSAME*
strides
2
block3_conv2/Conv2D?
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block3_conv2/BiasAdd/ReadVariableOp?
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?2
block3_conv2/BiasAdd?
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????KK?2
block3_conv2/Relu?
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block3_conv3/Conv2D/ReadVariableOp?
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?*
paddingSAME*
strides
2
block3_conv3/Conv2D?
#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block3_conv3/BiasAdd/ReadVariableOp?
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?2
block3_conv3/BiasAdd?
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????KK?2
block3_conv3/Relu?
block3_pool/MaxPoolMaxPoolblock3_conv3/Relu:activations:0*0
_output_shapes
:?????????%%?*
ksize
*
paddingVALID*
strides
2
block3_pool/MaxPool?
"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block4_conv1/Conv2D/ReadVariableOp?
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?*
paddingSAME*
strides
2
block4_conv1/Conv2D?
#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block4_conv1/BiasAdd/ReadVariableOp?
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?2
block4_conv1/BiasAdd?
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????%%?2
block4_conv1/Relu?
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block4_conv2/Conv2D/ReadVariableOp?
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?*
paddingSAME*
strides
2
block4_conv2/Conv2D?
#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block4_conv2/BiasAdd/ReadVariableOp?
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?2
block4_conv2/BiasAdd?
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????%%?2
block4_conv2/Relu?
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block4_conv3/Conv2D/ReadVariableOp?
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?*
paddingSAME*
strides
2
block4_conv3/Conv2D?
#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block4_conv3/BiasAdd/ReadVariableOp?
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?2
block4_conv3/BiasAdd?
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????%%?2
block4_conv3/Relu?
block4_pool/MaxPoolMaxPoolblock4_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
block4_pool/MaxPool?
"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block5_conv1/Conv2D/ReadVariableOp?
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
block5_conv1/Conv2D?
#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block5_conv1/BiasAdd/ReadVariableOp?
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
block5_conv1/BiasAdd?
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block5_conv1/Relu?
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block5_conv2/Conv2D/ReadVariableOp?
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
block5_conv2/Conv2D?
#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block5_conv2/BiasAdd/ReadVariableOp?
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
block5_conv2/BiasAdd?
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block5_conv2/Relu?
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block5_conv3/Conv2D/ReadVariableOp?
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
block5_conv3/Conv2D?
#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block5_conv3/BiasAdd/ReadVariableOp?
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
block5_conv3/BiasAdd?
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block5_conv3/Relu?
block5_pool/MaxPoolMaxPoolblock5_conv3/Relu:activations:0*0
_output_shapes
:?????????		?*
ksize
*
paddingVALID*
strides
2
block5_pool/MaxPool?
IdentityIdentityblock5_pool/MaxPool:output:0$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2J
#block5_conv2/BiasAdd/ReadVariableOp#block5_conv2/BiasAdd/ReadVariableOp2H
"block5_conv2/Conv2D/ReadVariableOp"block5_conv2/Conv2D/ReadVariableOp2J
#block5_conv3/BiasAdd/ReadVariableOp#block5_conv3/BiasAdd/ReadVariableOp2H
"block5_conv3/Conv2D/ReadVariableOp"block5_conv3/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
H__inference_block3_conv2_layer_call_and_return_conditional_losses_109196

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????KK?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????KK?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????KK?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????KK?
 
_user_specified_nameinputs
??
?'
D__inference_model_28_layer_call_and_return_conditional_losses_108401

inputsK
1vgg16_block1_conv1_conv2d_readvariableop_resource:@@
2vgg16_block1_conv1_biasadd_readvariableop_resource:@K
1vgg16_block1_conv2_conv2d_readvariableop_resource:@@@
2vgg16_block1_conv2_biasadd_readvariableop_resource:@L
1vgg16_block2_conv1_conv2d_readvariableop_resource:@?A
2vgg16_block2_conv1_biasadd_readvariableop_resource:	?M
1vgg16_block2_conv2_conv2d_readvariableop_resource:??A
2vgg16_block2_conv2_biasadd_readvariableop_resource:	?M
1vgg16_block3_conv1_conv2d_readvariableop_resource:??A
2vgg16_block3_conv1_biasadd_readvariableop_resource:	?M
1vgg16_block3_conv2_conv2d_readvariableop_resource:??A
2vgg16_block3_conv2_biasadd_readvariableop_resource:	?M
1vgg16_block3_conv3_conv2d_readvariableop_resource:??A
2vgg16_block3_conv3_biasadd_readvariableop_resource:	?M
1vgg16_block4_conv1_conv2d_readvariableop_resource:??A
2vgg16_block4_conv1_biasadd_readvariableop_resource:	?M
1vgg16_block4_conv2_conv2d_readvariableop_resource:??A
2vgg16_block4_conv2_biasadd_readvariableop_resource:	?M
1vgg16_block4_conv3_conv2d_readvariableop_resource:??A
2vgg16_block4_conv3_biasadd_readvariableop_resource:	?M
1vgg16_block5_conv1_conv2d_readvariableop_resource:??A
2vgg16_block5_conv1_biasadd_readvariableop_resource:	?M
1vgg16_block5_conv2_conv2d_readvariableop_resource:??A
2vgg16_block5_conv2_biasadd_readvariableop_resource:	?M
1vgg16_block5_conv3_conv2d_readvariableop_resource:??A
2vgg16_block5_conv3_biasadd_readvariableop_resource:	?D
0watermark_hidden1_matmul_readvariableop_resource:
??@?
1watermark_hidden1_biasadd_readvariableop_resource:@B
0watermark_hidden2_matmul_readvariableop_resource:@ ?
1watermark_hidden2_biasadd_readvariableop_resource: B
.cartoon_hidden1_matmul_readvariableop_resource:
??@=
/cartoon_hidden1_biasadd_readvariableop_resource:@B
0watermark_hidden3_matmul_readvariableop_resource: ?
1watermark_hidden3_biasadd_readvariableop_resource:@
.cartoon_hidden2_matmul_readvariableop_resource:@=
/cartoon_hidden2_biasadd_readvariableop_resource:B
0watermark_hidden4_matmul_readvariableop_resource:
?
1watermark_hidden4_biasadd_readvariableop_resource:
@
.cartoon_hidden3_matmul_readvariableop_resource:
=
/cartoon_hidden3_biasadd_readvariableop_resource:
A
/watermark_output_matmul_readvariableop_resource:
>
0watermark_output_biasadd_readvariableop_resource:?
-cartoon_output_matmul_readvariableop_resource:
<
.cartoon_output_biasadd_readvariableop_resource:
identity

identity_1??&cartoon_hidden1/BiasAdd/ReadVariableOp?%cartoon_hidden1/MatMul/ReadVariableOp?&cartoon_hidden2/BiasAdd/ReadVariableOp?%cartoon_hidden2/MatMul/ReadVariableOp?&cartoon_hidden3/BiasAdd/ReadVariableOp?%cartoon_hidden3/MatMul/ReadVariableOp?%cartoon_output/BiasAdd/ReadVariableOp?$cartoon_output/MatMul/ReadVariableOp?)vgg16/block1_conv1/BiasAdd/ReadVariableOp?(vgg16/block1_conv1/Conv2D/ReadVariableOp?)vgg16/block1_conv2/BiasAdd/ReadVariableOp?(vgg16/block1_conv2/Conv2D/ReadVariableOp?)vgg16/block2_conv1/BiasAdd/ReadVariableOp?(vgg16/block2_conv1/Conv2D/ReadVariableOp?)vgg16/block2_conv2/BiasAdd/ReadVariableOp?(vgg16/block2_conv2/Conv2D/ReadVariableOp?)vgg16/block3_conv1/BiasAdd/ReadVariableOp?(vgg16/block3_conv1/Conv2D/ReadVariableOp?)vgg16/block3_conv2/BiasAdd/ReadVariableOp?(vgg16/block3_conv2/Conv2D/ReadVariableOp?)vgg16/block3_conv3/BiasAdd/ReadVariableOp?(vgg16/block3_conv3/Conv2D/ReadVariableOp?)vgg16/block4_conv1/BiasAdd/ReadVariableOp?(vgg16/block4_conv1/Conv2D/ReadVariableOp?)vgg16/block4_conv2/BiasAdd/ReadVariableOp?(vgg16/block4_conv2/Conv2D/ReadVariableOp?)vgg16/block4_conv3/BiasAdd/ReadVariableOp?(vgg16/block4_conv3/Conv2D/ReadVariableOp?)vgg16/block5_conv1/BiasAdd/ReadVariableOp?(vgg16/block5_conv1/Conv2D/ReadVariableOp?)vgg16/block5_conv2/BiasAdd/ReadVariableOp?(vgg16/block5_conv2/Conv2D/ReadVariableOp?)vgg16/block5_conv3/BiasAdd/ReadVariableOp?(vgg16/block5_conv3/Conv2D/ReadVariableOp?(watermark_hidden1/BiasAdd/ReadVariableOp?'watermark_hidden1/MatMul/ReadVariableOp?(watermark_hidden2/BiasAdd/ReadVariableOp?'watermark_hidden2/MatMul/ReadVariableOp?(watermark_hidden3/BiasAdd/ReadVariableOp?'watermark_hidden3/MatMul/ReadVariableOp?(watermark_hidden4/BiasAdd/ReadVariableOp?'watermark_hidden4/MatMul/ReadVariableOp?'watermark_output/BiasAdd/ReadVariableOp?&watermark_output/MatMul/ReadVariableOpo
rescaling_24/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???;2
rescaling_24/Cast/xs
rescaling_24/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling_24/Cast_1/x?
rescaling_24/mulMulinputsrescaling_24/Cast/x:output:0*
T0*1
_output_shapes
:???????????2
rescaling_24/mul?
rescaling_24/addAddV2rescaling_24/mul:z:0rescaling_24/Cast_1/x:output:0*
T0*1
_output_shapes
:???????????2
rescaling_24/add?
(vgg16/block1_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02*
(vgg16/block1_conv1/Conv2D/ReadVariableOp?
vgg16/block1_conv1/Conv2DConv2Drescaling_24/add:z:00vgg16/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
vgg16/block1_conv1/Conv2D?
)vgg16/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)vgg16/block1_conv1/BiasAdd/ReadVariableOp?
vgg16/block1_conv1/BiasAddBiasAdd"vgg16/block1_conv1/Conv2D:output:01vgg16/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
vgg16/block1_conv1/BiasAdd?
vgg16/block1_conv1/ReluRelu#vgg16/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
vgg16/block1_conv1/Relu?
(vgg16/block1_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02*
(vgg16/block1_conv2/Conv2D/ReadVariableOp?
vgg16/block1_conv2/Conv2DConv2D%vgg16/block1_conv1/Relu:activations:00vgg16/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
vgg16/block1_conv2/Conv2D?
)vgg16/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)vgg16/block1_conv2/BiasAdd/ReadVariableOp?
vgg16/block1_conv2/BiasAddBiasAdd"vgg16/block1_conv2/Conv2D:output:01vgg16/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
vgg16/block1_conv2/BiasAdd?
vgg16/block1_conv2/ReluRelu#vgg16/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
vgg16/block1_conv2/Relu?
vgg16/block1_pool/MaxPoolMaxPool%vgg16/block1_conv2/Relu:activations:0*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
2
vgg16/block1_pool/MaxPool?
(vgg16/block2_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02*
(vgg16/block2_conv1/Conv2D/ReadVariableOp?
vgg16/block2_conv1/Conv2DConv2D"vgg16/block1_pool/MaxPool:output:00vgg16/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
vgg16/block2_conv1/Conv2D?
)vgg16/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block2_conv1/BiasAdd/ReadVariableOp?
vgg16/block2_conv1/BiasAddBiasAdd"vgg16/block2_conv1/Conv2D:output:01vgg16/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
vgg16/block2_conv1/BiasAdd?
vgg16/block2_conv1/ReluRelu#vgg16/block2_conv1/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
vgg16/block2_conv1/Relu?
(vgg16/block2_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block2_conv2/Conv2D/ReadVariableOp?
vgg16/block2_conv2/Conv2DConv2D%vgg16/block2_conv1/Relu:activations:00vgg16/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
vgg16/block2_conv2/Conv2D?
)vgg16/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block2_conv2/BiasAdd/ReadVariableOp?
vgg16/block2_conv2/BiasAddBiasAdd"vgg16/block2_conv2/Conv2D:output:01vgg16/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
vgg16/block2_conv2/BiasAdd?
vgg16/block2_conv2/ReluRelu#vgg16/block2_conv2/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
vgg16/block2_conv2/Relu?
vgg16/block2_pool/MaxPoolMaxPool%vgg16/block2_conv2/Relu:activations:0*0
_output_shapes
:?????????KK?*
ksize
*
paddingVALID*
strides
2
vgg16/block2_pool/MaxPool?
(vgg16/block3_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block3_conv1/Conv2D/ReadVariableOp?
vgg16/block3_conv1/Conv2DConv2D"vgg16/block2_pool/MaxPool:output:00vgg16/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?*
paddingSAME*
strides
2
vgg16/block3_conv1/Conv2D?
)vgg16/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block3_conv1/BiasAdd/ReadVariableOp?
vgg16/block3_conv1/BiasAddBiasAdd"vgg16/block3_conv1/Conv2D:output:01vgg16/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?2
vgg16/block3_conv1/BiasAdd?
vgg16/block3_conv1/ReluRelu#vgg16/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????KK?2
vgg16/block3_conv1/Relu?
(vgg16/block3_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block3_conv2/Conv2D/ReadVariableOp?
vgg16/block3_conv2/Conv2DConv2D%vgg16/block3_conv1/Relu:activations:00vgg16/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?*
paddingSAME*
strides
2
vgg16/block3_conv2/Conv2D?
)vgg16/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block3_conv2/BiasAdd/ReadVariableOp?
vgg16/block3_conv2/BiasAddBiasAdd"vgg16/block3_conv2/Conv2D:output:01vgg16/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?2
vgg16/block3_conv2/BiasAdd?
vgg16/block3_conv2/ReluRelu#vgg16/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????KK?2
vgg16/block3_conv2/Relu?
(vgg16/block3_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg16_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block3_conv3/Conv2D/ReadVariableOp?
vgg16/block3_conv3/Conv2DConv2D%vgg16/block3_conv2/Relu:activations:00vgg16/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?*
paddingSAME*
strides
2
vgg16/block3_conv3/Conv2D?
)vgg16/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block3_conv3/BiasAdd/ReadVariableOp?
vgg16/block3_conv3/BiasAddBiasAdd"vgg16/block3_conv3/Conv2D:output:01vgg16/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?2
vgg16/block3_conv3/BiasAdd?
vgg16/block3_conv3/ReluRelu#vgg16/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????KK?2
vgg16/block3_conv3/Relu?
vgg16/block3_pool/MaxPoolMaxPool%vgg16/block3_conv3/Relu:activations:0*0
_output_shapes
:?????????%%?*
ksize
*
paddingVALID*
strides
2
vgg16/block3_pool/MaxPool?
(vgg16/block4_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block4_conv1/Conv2D/ReadVariableOp?
vgg16/block4_conv1/Conv2DConv2D"vgg16/block3_pool/MaxPool:output:00vgg16/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?*
paddingSAME*
strides
2
vgg16/block4_conv1/Conv2D?
)vgg16/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block4_conv1/BiasAdd/ReadVariableOp?
vgg16/block4_conv1/BiasAddBiasAdd"vgg16/block4_conv1/Conv2D:output:01vgg16/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?2
vgg16/block4_conv1/BiasAdd?
vgg16/block4_conv1/ReluRelu#vgg16/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????%%?2
vgg16/block4_conv1/Relu?
(vgg16/block4_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block4_conv2/Conv2D/ReadVariableOp?
vgg16/block4_conv2/Conv2DConv2D%vgg16/block4_conv1/Relu:activations:00vgg16/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?*
paddingSAME*
strides
2
vgg16/block4_conv2/Conv2D?
)vgg16/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block4_conv2/BiasAdd/ReadVariableOp?
vgg16/block4_conv2/BiasAddBiasAdd"vgg16/block4_conv2/Conv2D:output:01vgg16/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?2
vgg16/block4_conv2/BiasAdd?
vgg16/block4_conv2/ReluRelu#vgg16/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????%%?2
vgg16/block4_conv2/Relu?
(vgg16/block4_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg16_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block4_conv3/Conv2D/ReadVariableOp?
vgg16/block4_conv3/Conv2DConv2D%vgg16/block4_conv2/Relu:activations:00vgg16/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?*
paddingSAME*
strides
2
vgg16/block4_conv3/Conv2D?
)vgg16/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block4_conv3/BiasAdd/ReadVariableOp?
vgg16/block4_conv3/BiasAddBiasAdd"vgg16/block4_conv3/Conv2D:output:01vgg16/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?2
vgg16/block4_conv3/BiasAdd?
vgg16/block4_conv3/ReluRelu#vgg16/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????%%?2
vgg16/block4_conv3/Relu?
vgg16/block4_pool/MaxPoolMaxPool%vgg16/block4_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
vgg16/block4_pool/MaxPool?
(vgg16/block5_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block5_conv1/Conv2D/ReadVariableOp?
vgg16/block5_conv1/Conv2DConv2D"vgg16/block4_pool/MaxPool:output:00vgg16/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
vgg16/block5_conv1/Conv2D?
)vgg16/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block5_conv1/BiasAdd/ReadVariableOp?
vgg16/block5_conv1/BiasAddBiasAdd"vgg16/block5_conv1/Conv2D:output:01vgg16/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
vgg16/block5_conv1/BiasAdd?
vgg16/block5_conv1/ReluRelu#vgg16/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
vgg16/block5_conv1/Relu?
(vgg16/block5_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block5_conv2/Conv2D/ReadVariableOp?
vgg16/block5_conv2/Conv2DConv2D%vgg16/block5_conv1/Relu:activations:00vgg16/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
vgg16/block5_conv2/Conv2D?
)vgg16/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block5_conv2/BiasAdd/ReadVariableOp?
vgg16/block5_conv2/BiasAddBiasAdd"vgg16/block5_conv2/Conv2D:output:01vgg16/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
vgg16/block5_conv2/BiasAdd?
vgg16/block5_conv2/ReluRelu#vgg16/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
vgg16/block5_conv2/Relu?
(vgg16/block5_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg16_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(vgg16/block5_conv3/Conv2D/ReadVariableOp?
vgg16/block5_conv3/Conv2DConv2D%vgg16/block5_conv2/Relu:activations:00vgg16/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
vgg16/block5_conv3/Conv2D?
)vgg16/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)vgg16/block5_conv3/BiasAdd/ReadVariableOp?
vgg16/block5_conv3/BiasAddBiasAdd"vgg16/block5_conv3/Conv2D:output:01vgg16/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
vgg16/block5_conv3/BiasAdd?
vgg16/block5_conv3/ReluRelu#vgg16/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
vgg16/block5_conv3/Relu?
vgg16/block5_pool/MaxPoolMaxPool%vgg16/block5_conv3/Relu:activations:0*0
_output_shapes
:?????????		?*
ksize
*
paddingVALID*
strides
2
vgg16/block5_pool/MaxPoolu
flatten_28/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ?  2
flatten_28/Const?
flatten_28/ReshapeReshape"vgg16/block5_pool/MaxPool:output:0flatten_28/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_28/Reshape?
'watermark_hidden1/MatMul/ReadVariableOpReadVariableOp0watermark_hidden1_matmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype02)
'watermark_hidden1/MatMul/ReadVariableOp?
watermark_hidden1/MatMulMatMulflatten_28/Reshape:output:0/watermark_hidden1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
watermark_hidden1/MatMul?
(watermark_hidden1/BiasAdd/ReadVariableOpReadVariableOp1watermark_hidden1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(watermark_hidden1/BiasAdd/ReadVariableOp?
watermark_hidden1/BiasAddBiasAdd"watermark_hidden1/MatMul:product:00watermark_hidden1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
watermark_hidden1/BiasAdd?
watermark_hidden1/ReluRelu"watermark_hidden1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
watermark_hidden1/Relu?
'watermark_hidden2/MatMul/ReadVariableOpReadVariableOp0watermark_hidden2_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02)
'watermark_hidden2/MatMul/ReadVariableOp?
watermark_hidden2/MatMulMatMul$watermark_hidden1/Relu:activations:0/watermark_hidden2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
watermark_hidden2/MatMul?
(watermark_hidden2/BiasAdd/ReadVariableOpReadVariableOp1watermark_hidden2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(watermark_hidden2/BiasAdd/ReadVariableOp?
watermark_hidden2/BiasAddBiasAdd"watermark_hidden2/MatMul:product:00watermark_hidden2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
watermark_hidden2/BiasAdd?
watermark_hidden2/ReluRelu"watermark_hidden2/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
watermark_hidden2/Relu?
%cartoon_hidden1/MatMul/ReadVariableOpReadVariableOp.cartoon_hidden1_matmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype02'
%cartoon_hidden1/MatMul/ReadVariableOp?
cartoon_hidden1/MatMulMatMulflatten_28/Reshape:output:0-cartoon_hidden1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
cartoon_hidden1/MatMul?
&cartoon_hidden1/BiasAdd/ReadVariableOpReadVariableOp/cartoon_hidden1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&cartoon_hidden1/BiasAdd/ReadVariableOp?
cartoon_hidden1/BiasAddBiasAdd cartoon_hidden1/MatMul:product:0.cartoon_hidden1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
cartoon_hidden1/BiasAdd?
cartoon_hidden1/ReluRelu cartoon_hidden1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
cartoon_hidden1/Relu?
'watermark_hidden3/MatMul/ReadVariableOpReadVariableOp0watermark_hidden3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02)
'watermark_hidden3/MatMul/ReadVariableOp?
watermark_hidden3/MatMulMatMul$watermark_hidden2/Relu:activations:0/watermark_hidden3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
watermark_hidden3/MatMul?
(watermark_hidden3/BiasAdd/ReadVariableOpReadVariableOp1watermark_hidden3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(watermark_hidden3/BiasAdd/ReadVariableOp?
watermark_hidden3/BiasAddBiasAdd"watermark_hidden3/MatMul:product:00watermark_hidden3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
watermark_hidden3/BiasAdd?
watermark_hidden3/ReluRelu"watermark_hidden3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
watermark_hidden3/Relu?
%cartoon_hidden2/MatMul/ReadVariableOpReadVariableOp.cartoon_hidden2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02'
%cartoon_hidden2/MatMul/ReadVariableOp?
cartoon_hidden2/MatMulMatMul"cartoon_hidden1/Relu:activations:0-cartoon_hidden2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
cartoon_hidden2/MatMul?
&cartoon_hidden2/BiasAdd/ReadVariableOpReadVariableOp/cartoon_hidden2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&cartoon_hidden2/BiasAdd/ReadVariableOp?
cartoon_hidden2/BiasAddBiasAdd cartoon_hidden2/MatMul:product:0.cartoon_hidden2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
cartoon_hidden2/BiasAdd?
cartoon_hidden2/ReluRelu cartoon_hidden2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
cartoon_hidden2/Relu?
'watermark_hidden4/MatMul/ReadVariableOpReadVariableOp0watermark_hidden4_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02)
'watermark_hidden4/MatMul/ReadVariableOp?
watermark_hidden4/MatMulMatMul$watermark_hidden3/Relu:activations:0/watermark_hidden4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
watermark_hidden4/MatMul?
(watermark_hidden4/BiasAdd/ReadVariableOpReadVariableOp1watermark_hidden4_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02*
(watermark_hidden4/BiasAdd/ReadVariableOp?
watermark_hidden4/BiasAddBiasAdd"watermark_hidden4/MatMul:product:00watermark_hidden4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
watermark_hidden4/BiasAdd?
watermark_hidden4/ReluRelu"watermark_hidden4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
watermark_hidden4/Relu?
%cartoon_hidden3/MatMul/ReadVariableOpReadVariableOp.cartoon_hidden3_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02'
%cartoon_hidden3/MatMul/ReadVariableOp?
cartoon_hidden3/MatMulMatMul"cartoon_hidden2/Relu:activations:0-cartoon_hidden3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
cartoon_hidden3/MatMul?
&cartoon_hidden3/BiasAdd/ReadVariableOpReadVariableOp/cartoon_hidden3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02(
&cartoon_hidden3/BiasAdd/ReadVariableOp?
cartoon_hidden3/BiasAddBiasAdd cartoon_hidden3/MatMul:product:0.cartoon_hidden3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
cartoon_hidden3/BiasAdd?
cartoon_hidden3/ReluRelu cartoon_hidden3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
cartoon_hidden3/Relu?
&watermark_output/MatMul/ReadVariableOpReadVariableOp/watermark_output_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02(
&watermark_output/MatMul/ReadVariableOp?
watermark_output/MatMulMatMul$watermark_hidden4/Relu:activations:0.watermark_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
watermark_output/MatMul?
'watermark_output/BiasAdd/ReadVariableOpReadVariableOp0watermark_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'watermark_output/BiasAdd/ReadVariableOp?
watermark_output/BiasAddBiasAdd!watermark_output/MatMul:product:0/watermark_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
watermark_output/BiasAdd?
watermark_output/SigmoidSigmoid!watermark_output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
watermark_output/Sigmoid?
$cartoon_output/MatMul/ReadVariableOpReadVariableOp-cartoon_output_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02&
$cartoon_output/MatMul/ReadVariableOp?
cartoon_output/MatMulMatMul"cartoon_hidden3/Relu:activations:0,cartoon_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
cartoon_output/MatMul?
%cartoon_output/BiasAdd/ReadVariableOpReadVariableOp.cartoon_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%cartoon_output/BiasAdd/ReadVariableOp?
cartoon_output/BiasAddBiasAddcartoon_output/MatMul:product:0-cartoon_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
cartoon_output/BiasAdd?
cartoon_output/SigmoidSigmoidcartoon_output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
cartoon_output/Sigmoid?
IdentityIdentitycartoon_output/Sigmoid:y:0'^cartoon_hidden1/BiasAdd/ReadVariableOp&^cartoon_hidden1/MatMul/ReadVariableOp'^cartoon_hidden2/BiasAdd/ReadVariableOp&^cartoon_hidden2/MatMul/ReadVariableOp'^cartoon_hidden3/BiasAdd/ReadVariableOp&^cartoon_hidden3/MatMul/ReadVariableOp&^cartoon_output/BiasAdd/ReadVariableOp%^cartoon_output/MatMul/ReadVariableOp*^vgg16/block1_conv1/BiasAdd/ReadVariableOp)^vgg16/block1_conv1/Conv2D/ReadVariableOp*^vgg16/block1_conv2/BiasAdd/ReadVariableOp)^vgg16/block1_conv2/Conv2D/ReadVariableOp*^vgg16/block2_conv1/BiasAdd/ReadVariableOp)^vgg16/block2_conv1/Conv2D/ReadVariableOp*^vgg16/block2_conv2/BiasAdd/ReadVariableOp)^vgg16/block2_conv2/Conv2D/ReadVariableOp*^vgg16/block3_conv1/BiasAdd/ReadVariableOp)^vgg16/block3_conv1/Conv2D/ReadVariableOp*^vgg16/block3_conv2/BiasAdd/ReadVariableOp)^vgg16/block3_conv2/Conv2D/ReadVariableOp*^vgg16/block3_conv3/BiasAdd/ReadVariableOp)^vgg16/block3_conv3/Conv2D/ReadVariableOp*^vgg16/block4_conv1/BiasAdd/ReadVariableOp)^vgg16/block4_conv1/Conv2D/ReadVariableOp*^vgg16/block4_conv2/BiasAdd/ReadVariableOp)^vgg16/block4_conv2/Conv2D/ReadVariableOp*^vgg16/block4_conv3/BiasAdd/ReadVariableOp)^vgg16/block4_conv3/Conv2D/ReadVariableOp*^vgg16/block5_conv1/BiasAdd/ReadVariableOp)^vgg16/block5_conv1/Conv2D/ReadVariableOp*^vgg16/block5_conv2/BiasAdd/ReadVariableOp)^vgg16/block5_conv2/Conv2D/ReadVariableOp*^vgg16/block5_conv3/BiasAdd/ReadVariableOp)^vgg16/block5_conv3/Conv2D/ReadVariableOp)^watermark_hidden1/BiasAdd/ReadVariableOp(^watermark_hidden1/MatMul/ReadVariableOp)^watermark_hidden2/BiasAdd/ReadVariableOp(^watermark_hidden2/MatMul/ReadVariableOp)^watermark_hidden3/BiasAdd/ReadVariableOp(^watermark_hidden3/MatMul/ReadVariableOp)^watermark_hidden4/BiasAdd/ReadVariableOp(^watermark_hidden4/MatMul/ReadVariableOp(^watermark_output/BiasAdd/ReadVariableOp'^watermark_output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitywatermark_output/Sigmoid:y:0'^cartoon_hidden1/BiasAdd/ReadVariableOp&^cartoon_hidden1/MatMul/ReadVariableOp'^cartoon_hidden2/BiasAdd/ReadVariableOp&^cartoon_hidden2/MatMul/ReadVariableOp'^cartoon_hidden3/BiasAdd/ReadVariableOp&^cartoon_hidden3/MatMul/ReadVariableOp&^cartoon_output/BiasAdd/ReadVariableOp%^cartoon_output/MatMul/ReadVariableOp*^vgg16/block1_conv1/BiasAdd/ReadVariableOp)^vgg16/block1_conv1/Conv2D/ReadVariableOp*^vgg16/block1_conv2/BiasAdd/ReadVariableOp)^vgg16/block1_conv2/Conv2D/ReadVariableOp*^vgg16/block2_conv1/BiasAdd/ReadVariableOp)^vgg16/block2_conv1/Conv2D/ReadVariableOp*^vgg16/block2_conv2/BiasAdd/ReadVariableOp)^vgg16/block2_conv2/Conv2D/ReadVariableOp*^vgg16/block3_conv1/BiasAdd/ReadVariableOp)^vgg16/block3_conv1/Conv2D/ReadVariableOp*^vgg16/block3_conv2/BiasAdd/ReadVariableOp)^vgg16/block3_conv2/Conv2D/ReadVariableOp*^vgg16/block3_conv3/BiasAdd/ReadVariableOp)^vgg16/block3_conv3/Conv2D/ReadVariableOp*^vgg16/block4_conv1/BiasAdd/ReadVariableOp)^vgg16/block4_conv1/Conv2D/ReadVariableOp*^vgg16/block4_conv2/BiasAdd/ReadVariableOp)^vgg16/block4_conv2/Conv2D/ReadVariableOp*^vgg16/block4_conv3/BiasAdd/ReadVariableOp)^vgg16/block4_conv3/Conv2D/ReadVariableOp*^vgg16/block5_conv1/BiasAdd/ReadVariableOp)^vgg16/block5_conv1/Conv2D/ReadVariableOp*^vgg16/block5_conv2/BiasAdd/ReadVariableOp)^vgg16/block5_conv2/Conv2D/ReadVariableOp*^vgg16/block5_conv3/BiasAdd/ReadVariableOp)^vgg16/block5_conv3/Conv2D/ReadVariableOp)^watermark_hidden1/BiasAdd/ReadVariableOp(^watermark_hidden1/MatMul/ReadVariableOp)^watermark_hidden2/BiasAdd/ReadVariableOp(^watermark_hidden2/MatMul/ReadVariableOp)^watermark_hidden3/BiasAdd/ReadVariableOp(^watermark_hidden3/MatMul/ReadVariableOp)^watermark_hidden4/BiasAdd/ReadVariableOp(^watermark_hidden4/MatMul/ReadVariableOp(^watermark_output/BiasAdd/ReadVariableOp'^watermark_output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesw
u:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&cartoon_hidden1/BiasAdd/ReadVariableOp&cartoon_hidden1/BiasAdd/ReadVariableOp2N
%cartoon_hidden1/MatMul/ReadVariableOp%cartoon_hidden1/MatMul/ReadVariableOp2P
&cartoon_hidden2/BiasAdd/ReadVariableOp&cartoon_hidden2/BiasAdd/ReadVariableOp2N
%cartoon_hidden2/MatMul/ReadVariableOp%cartoon_hidden2/MatMul/ReadVariableOp2P
&cartoon_hidden3/BiasAdd/ReadVariableOp&cartoon_hidden3/BiasAdd/ReadVariableOp2N
%cartoon_hidden3/MatMul/ReadVariableOp%cartoon_hidden3/MatMul/ReadVariableOp2N
%cartoon_output/BiasAdd/ReadVariableOp%cartoon_output/BiasAdd/ReadVariableOp2L
$cartoon_output/MatMul/ReadVariableOp$cartoon_output/MatMul/ReadVariableOp2V
)vgg16/block1_conv1/BiasAdd/ReadVariableOp)vgg16/block1_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block1_conv1/Conv2D/ReadVariableOp(vgg16/block1_conv1/Conv2D/ReadVariableOp2V
)vgg16/block1_conv2/BiasAdd/ReadVariableOp)vgg16/block1_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block1_conv2/Conv2D/ReadVariableOp(vgg16/block1_conv2/Conv2D/ReadVariableOp2V
)vgg16/block2_conv1/BiasAdd/ReadVariableOp)vgg16/block2_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block2_conv1/Conv2D/ReadVariableOp(vgg16/block2_conv1/Conv2D/ReadVariableOp2V
)vgg16/block2_conv2/BiasAdd/ReadVariableOp)vgg16/block2_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block2_conv2/Conv2D/ReadVariableOp(vgg16/block2_conv2/Conv2D/ReadVariableOp2V
)vgg16/block3_conv1/BiasAdd/ReadVariableOp)vgg16/block3_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block3_conv1/Conv2D/ReadVariableOp(vgg16/block3_conv1/Conv2D/ReadVariableOp2V
)vgg16/block3_conv2/BiasAdd/ReadVariableOp)vgg16/block3_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block3_conv2/Conv2D/ReadVariableOp(vgg16/block3_conv2/Conv2D/ReadVariableOp2V
)vgg16/block3_conv3/BiasAdd/ReadVariableOp)vgg16/block3_conv3/BiasAdd/ReadVariableOp2T
(vgg16/block3_conv3/Conv2D/ReadVariableOp(vgg16/block3_conv3/Conv2D/ReadVariableOp2V
)vgg16/block4_conv1/BiasAdd/ReadVariableOp)vgg16/block4_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block4_conv1/Conv2D/ReadVariableOp(vgg16/block4_conv1/Conv2D/ReadVariableOp2V
)vgg16/block4_conv2/BiasAdd/ReadVariableOp)vgg16/block4_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block4_conv2/Conv2D/ReadVariableOp(vgg16/block4_conv2/Conv2D/ReadVariableOp2V
)vgg16/block4_conv3/BiasAdd/ReadVariableOp)vgg16/block4_conv3/BiasAdd/ReadVariableOp2T
(vgg16/block4_conv3/Conv2D/ReadVariableOp(vgg16/block4_conv3/Conv2D/ReadVariableOp2V
)vgg16/block5_conv1/BiasAdd/ReadVariableOp)vgg16/block5_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block5_conv1/Conv2D/ReadVariableOp(vgg16/block5_conv1/Conv2D/ReadVariableOp2V
)vgg16/block5_conv2/BiasAdd/ReadVariableOp)vgg16/block5_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block5_conv2/Conv2D/ReadVariableOp(vgg16/block5_conv2/Conv2D/ReadVariableOp2V
)vgg16/block5_conv3/BiasAdd/ReadVariableOp)vgg16/block5_conv3/BiasAdd/ReadVariableOp2T
(vgg16/block5_conv3/Conv2D/ReadVariableOp(vgg16/block5_conv3/Conv2D/ReadVariableOp2T
(watermark_hidden1/BiasAdd/ReadVariableOp(watermark_hidden1/BiasAdd/ReadVariableOp2R
'watermark_hidden1/MatMul/ReadVariableOp'watermark_hidden1/MatMul/ReadVariableOp2T
(watermark_hidden2/BiasAdd/ReadVariableOp(watermark_hidden2/BiasAdd/ReadVariableOp2R
'watermark_hidden2/MatMul/ReadVariableOp'watermark_hidden2/MatMul/ReadVariableOp2T
(watermark_hidden3/BiasAdd/ReadVariableOp(watermark_hidden3/BiasAdd/ReadVariableOp2R
'watermark_hidden3/MatMul/ReadVariableOp'watermark_hidden3/MatMul/ReadVariableOp2T
(watermark_hidden4/BiasAdd/ReadVariableOp(watermark_hidden4/BiasAdd/ReadVariableOp2R
'watermark_hidden4/MatMul/ReadVariableOp'watermark_hidden4/MatMul/ReadVariableOp2R
'watermark_output/BiasAdd/ReadVariableOp'watermark_output/BiasAdd/ReadVariableOp2P
&watermark_output/MatMul/ReadVariableOp&watermark_output/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
H
,__inference_block2_pool_layer_call_fn_106071

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block2_pool_layer_call_and_return_conditional_losses_1060652
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
H__inference_block4_conv2_layer_call_and_return_conditional_losses_106264

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????%%?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????%%?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????%%?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????%%?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????%%?
 
_user_specified_nameinputs
?

?
J__inference_cartoon_output_layer_call_and_return_conditional_losses_107138

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
-__inference_block3_conv2_layer_call_fn_109185

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????KK?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block3_conv2_layer_call_and_return_conditional_losses_1062122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????KK?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????KK?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????KK?
 
_user_specified_nameinputs
?
G
+__inference_flatten_28_layer_call_fn_108890

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_28_layer_call_and_return_conditional_losses_1069892
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????		?:X T
0
_output_shapes
:?????????		?
 
_user_specified_nameinputs
?

?
M__inference_watermark_hidden4_layer_call_and_return_conditional_losses_109036

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
H
,__inference_block1_pool_layer_call_fn_106059

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block1_pool_layer_call_and_return_conditional_losses_1060532
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
-__inference_block4_conv1_layer_call_fn_109225

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????%%?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block4_conv1_layer_call_and_return_conditional_losses_1062472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????%%?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????%%?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????%%?
 
_user_specified_nameinputs
?

?
K__inference_cartoon_hidden1_layer_call_and_return_conditional_losses_107036

inputs2
matmul_readvariableop_resource:
??@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
H
,__inference_block3_pool_layer_call_fn_106083

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_pool_layer_call_and_return_conditional_losses_1060772
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
&__inference_vgg16_layer_call_fn_108628

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_vgg16_layer_call_and_return_conditional_losses_1063412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
H
,__inference_block4_pool_layer_call_fn_106095

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_pool_layer_call_and_return_conditional_losses_1060892
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
H__inference_block5_conv2_layer_call_and_return_conditional_losses_106316

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
H__inference_block1_conv1_layer_call_and_return_conditional_losses_106125

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
-__inference_block1_conv2_layer_call_fn_109105

inputs!
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block1_conv2_layer_call_and_return_conditional_losses_1061422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
)__inference_model_28_layer_call_fn_108231

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?

unknown_25:
??@

unknown_26:@

unknown_27:@ 

unknown_28: 

unknown_29:
??@

unknown_30:@

unknown_31: 

unknown_32:

unknown_33:@

unknown_34:

unknown_35:


unknown_36:


unknown_37:


unknown_38:


unknown_39:


unknown_40:

unknown_41:


unknown_42:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_28_layer_call_and_return_conditional_losses_1075402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesw
u:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
K__inference_cartoon_hidden2_layer_call_and_return_conditional_losses_108976

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
K__inference_cartoon_hidden1_layer_call_and_return_conditional_losses_108936

inputs2
matmul_readvariableop_resource:
??@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
M__inference_watermark_hidden2_layer_call_and_return_conditional_losses_108956

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
H__inference_block3_conv2_layer_call_and_return_conditional_losses_106212

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????KK?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????KK?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????KK?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????KK?
 
_user_specified_nameinputs
?
?
)__inference_model_28_layer_call_fn_108136

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?

unknown_25:
??@

unknown_26:@

unknown_27:@ 

unknown_28: 

unknown_29:
??@

unknown_30:@

unknown_31: 

unknown_32:

unknown_33:@

unknown_34:

unknown_35:


unknown_36:


unknown_37:


unknown_38:


unknown_39:


unknown_40:

unknown_41:


unknown_42:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_28_layer_call_and_return_conditional_losses_1071462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesw
u:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
&__inference_vgg16_layer_call_fn_106771
input_1!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_vgg16_layer_call_and_return_conditional_losses_1066592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
0__inference_cartoon_hidden2_layer_call_fn_108965

inputs
unknown:@
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_cartoon_hidden2_layer_call_and_return_conditional_losses_1070702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
)__inference_model_28_layer_call_fn_107728
input_31!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?

unknown_25:
??@

unknown_26:@

unknown_27:@ 

unknown_28: 

unknown_29:
??@

unknown_30:@

unknown_31: 

unknown_32:

unknown_33:@

unknown_34:

unknown_35:


unknown_36:


unknown_37:


unknown_38:


unknown_39:


unknown_40:

unknown_41:


unknown_42:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_31unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_28_layer_call_and_return_conditional_losses_1075402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesw
u:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_31
?
c
G__inference_block4_pool_layer_call_and_return_conditional_losses_106089

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
-__inference_block4_conv3_layer_call_fn_109265

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????%%?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block4_conv3_layer_call_and_return_conditional_losses_1062812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????%%?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????%%?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????%%?
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_108041
input_31!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?

unknown_25:
??@

unknown_26:@

unknown_27:@ 

unknown_28: 

unknown_29:
??@

unknown_30:@

unknown_31: 

unknown_32:

unknown_33:@

unknown_34:

unknown_35:


unknown_36:


unknown_37:


unknown_38:


unknown_39:


unknown_40:

unknown_41:


unknown_42:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_31unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_1060472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesw
u:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_31
?
c
G__inference_block1_pool_layer_call_and_return_conditional_losses_106053

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
-__inference_block1_conv1_layer_call_fn_109085

inputs!
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block1_conv1_layer_call_and_return_conditional_losses_1061252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
H__inference_block3_conv1_layer_call_and_return_conditional_losses_109176

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????KK?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????KK?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????KK?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????KK?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????KK?
 
_user_specified_nameinputs
?\
?
A__inference_vgg16_layer_call_and_return_conditional_losses_106845
input_1-
block1_conv1_106774:@!
block1_conv1_106776:@-
block1_conv2_106779:@@!
block1_conv2_106781:@.
block2_conv1_106785:@?"
block2_conv1_106787:	?/
block2_conv2_106790:??"
block2_conv2_106792:	?/
block3_conv1_106796:??"
block3_conv1_106798:	?/
block3_conv2_106801:??"
block3_conv2_106803:	?/
block3_conv3_106806:??"
block3_conv3_106808:	?/
block4_conv1_106812:??"
block4_conv1_106814:	?/
block4_conv2_106817:??"
block4_conv2_106819:	?/
block4_conv3_106822:??"
block4_conv3_106824:	?/
block5_conv1_106828:??"
block5_conv1_106830:	?/
block5_conv2_106833:??"
block5_conv2_106835:	?/
block5_conv3_106838:??"
block5_conv3_106840:	?
identity??$block1_conv1/StatefulPartitionedCall?$block1_conv2/StatefulPartitionedCall?$block2_conv1/StatefulPartitionedCall?$block2_conv2/StatefulPartitionedCall?$block3_conv1/StatefulPartitionedCall?$block3_conv2/StatefulPartitionedCall?$block3_conv3/StatefulPartitionedCall?$block4_conv1/StatefulPartitionedCall?$block4_conv2/StatefulPartitionedCall?$block4_conv3/StatefulPartitionedCall?$block5_conv1/StatefulPartitionedCall?$block5_conv2/StatefulPartitionedCall?$block5_conv3/StatefulPartitionedCall?
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_1block1_conv1_106774block1_conv1_106776*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block1_conv1_layer_call_and_return_conditional_losses_1061252&
$block1_conv1/StatefulPartitionedCall?
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_106779block1_conv2_106781*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block1_conv2_layer_call_and_return_conditional_losses_1061422&
$block1_conv2/StatefulPartitionedCall?
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block1_pool_layer_call_and_return_conditional_losses_1060532
block1_pool/PartitionedCall?
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_106785block2_conv1_106787*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block2_conv1_layer_call_and_return_conditional_losses_1061602&
$block2_conv1/StatefulPartitionedCall?
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_106790block2_conv2_106792*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block2_conv2_layer_call_and_return_conditional_losses_1061772&
$block2_conv2/StatefulPartitionedCall?
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????KK?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block2_pool_layer_call_and_return_conditional_losses_1060652
block2_pool/PartitionedCall?
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_106796block3_conv1_106798*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????KK?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block3_conv1_layer_call_and_return_conditional_losses_1061952&
$block3_conv1/StatefulPartitionedCall?
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_106801block3_conv2_106803*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????KK?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block3_conv2_layer_call_and_return_conditional_losses_1062122&
$block3_conv2/StatefulPartitionedCall?
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_106806block3_conv3_106808*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????KK?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block3_conv3_layer_call_and_return_conditional_losses_1062292&
$block3_conv3/StatefulPartitionedCall?
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????%%?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_pool_layer_call_and_return_conditional_losses_1060772
block3_pool/PartitionedCall?
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_106812block4_conv1_106814*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????%%?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block4_conv1_layer_call_and_return_conditional_losses_1062472&
$block4_conv1/StatefulPartitionedCall?
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_106817block4_conv2_106819*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????%%?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block4_conv2_layer_call_and_return_conditional_losses_1062642&
$block4_conv2/StatefulPartitionedCall?
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_106822block4_conv3_106824*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????%%?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block4_conv3_layer_call_and_return_conditional_losses_1062812&
$block4_conv3/StatefulPartitionedCall?
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_pool_layer_call_and_return_conditional_losses_1060892
block4_pool/PartitionedCall?
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_106828block5_conv1_106830*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block5_conv1_layer_call_and_return_conditional_losses_1062992&
$block5_conv1/StatefulPartitionedCall?
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_106833block5_conv2_106835*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block5_conv2_layer_call_and_return_conditional_losses_1063162&
$block5_conv2/StatefulPartitionedCall?
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_106838block5_conv3_106840*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block5_conv3_layer_call_and_return_conditional_losses_1063332&
$block5_conv3/StatefulPartitionedCall?
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_pool_layer_call_and_return_conditional_losses_1061012
block5_pool/PartitionedCall?
IdentityIdentity$block5_pool/PartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
-__inference_block2_conv1_layer_call_fn_109125

inputs"
unknown:@?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block2_conv1_layer_call_and_return_conditional_losses_1061602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?R
?
D__inference_model_28_layer_call_and_return_conditional_losses_107540

inputs&
vgg16_107439:@
vgg16_107441:@&
vgg16_107443:@@
vgg16_107445:@'
vgg16_107447:@?
vgg16_107449:	?(
vgg16_107451:??
vgg16_107453:	?(
vgg16_107455:??
vgg16_107457:	?(
vgg16_107459:??
vgg16_107461:	?(
vgg16_107463:??
vgg16_107465:	?(
vgg16_107467:??
vgg16_107469:	?(
vgg16_107471:??
vgg16_107473:	?(
vgg16_107475:??
vgg16_107477:	?(
vgg16_107479:??
vgg16_107481:	?(
vgg16_107483:??
vgg16_107485:	?(
vgg16_107487:??
vgg16_107489:	?,
watermark_hidden1_107493:
??@&
watermark_hidden1_107495:@*
watermark_hidden2_107498:@ &
watermark_hidden2_107500: *
cartoon_hidden1_107503:
??@$
cartoon_hidden1_107505:@*
watermark_hidden3_107508: &
watermark_hidden3_107510:(
cartoon_hidden2_107513:@$
cartoon_hidden2_107515:*
watermark_hidden4_107518:
&
watermark_hidden4_107520:
(
cartoon_hidden3_107523:
$
cartoon_hidden3_107525:
)
watermark_output_107528:
%
watermark_output_107530:'
cartoon_output_107533:
#
cartoon_output_107535:
identity

identity_1??'cartoon_hidden1/StatefulPartitionedCall?'cartoon_hidden2/StatefulPartitionedCall?'cartoon_hidden3/StatefulPartitionedCall?&cartoon_output/StatefulPartitionedCall?vgg16/StatefulPartitionedCall?)watermark_hidden1/StatefulPartitionedCall?)watermark_hidden2/StatefulPartitionedCall?)watermark_hidden3/StatefulPartitionedCall?)watermark_hidden4/StatefulPartitionedCall?(watermark_output/StatefulPartitionedCallo
rescaling_24/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???;2
rescaling_24/Cast/xs
rescaling_24/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling_24/Cast_1/x?
rescaling_24/mulMulinputsrescaling_24/Cast/x:output:0*
T0*1
_output_shapes
:???????????2
rescaling_24/mul?
rescaling_24/addAddV2rescaling_24/mul:z:0rescaling_24/Cast_1/x:output:0*
T0*1
_output_shapes
:???????????2
rescaling_24/add?
vgg16/StatefulPartitionedCallStatefulPartitionedCallrescaling_24/add:z:0vgg16_107439vgg16_107441vgg16_107443vgg16_107445vgg16_107447vgg16_107449vgg16_107451vgg16_107453vgg16_107455vgg16_107457vgg16_107459vgg16_107461vgg16_107463vgg16_107465vgg16_107467vgg16_107469vgg16_107471vgg16_107473vgg16_107475vgg16_107477vgg16_107479vgg16_107481vgg16_107483vgg16_107485vgg16_107487vgg16_107489*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_vgg16_layer_call_and_return_conditional_losses_1066592
vgg16/StatefulPartitionedCall?
flatten_28/PartitionedCallPartitionedCall&vgg16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_28_layer_call_and_return_conditional_losses_1069892
flatten_28/PartitionedCall?
)watermark_hidden1/StatefulPartitionedCallStatefulPartitionedCall#flatten_28/PartitionedCall:output:0watermark_hidden1_107493watermark_hidden1_107495*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_watermark_hidden1_layer_call_and_return_conditional_losses_1070022+
)watermark_hidden1/StatefulPartitionedCall?
)watermark_hidden2/StatefulPartitionedCallStatefulPartitionedCall2watermark_hidden1/StatefulPartitionedCall:output:0watermark_hidden2_107498watermark_hidden2_107500*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_watermark_hidden2_layer_call_and_return_conditional_losses_1070192+
)watermark_hidden2/StatefulPartitionedCall?
'cartoon_hidden1/StatefulPartitionedCallStatefulPartitionedCall#flatten_28/PartitionedCall:output:0cartoon_hidden1_107503cartoon_hidden1_107505*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_cartoon_hidden1_layer_call_and_return_conditional_losses_1070362)
'cartoon_hidden1/StatefulPartitionedCall?
)watermark_hidden3/StatefulPartitionedCallStatefulPartitionedCall2watermark_hidden2/StatefulPartitionedCall:output:0watermark_hidden3_107508watermark_hidden3_107510*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_watermark_hidden3_layer_call_and_return_conditional_losses_1070532+
)watermark_hidden3/StatefulPartitionedCall?
'cartoon_hidden2/StatefulPartitionedCallStatefulPartitionedCall0cartoon_hidden1/StatefulPartitionedCall:output:0cartoon_hidden2_107513cartoon_hidden2_107515*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_cartoon_hidden2_layer_call_and_return_conditional_losses_1070702)
'cartoon_hidden2/StatefulPartitionedCall?
)watermark_hidden4/StatefulPartitionedCallStatefulPartitionedCall2watermark_hidden3/StatefulPartitionedCall:output:0watermark_hidden4_107518watermark_hidden4_107520*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_watermark_hidden4_layer_call_and_return_conditional_losses_1070872+
)watermark_hidden4/StatefulPartitionedCall?
'cartoon_hidden3/StatefulPartitionedCallStatefulPartitionedCall0cartoon_hidden2/StatefulPartitionedCall:output:0cartoon_hidden3_107523cartoon_hidden3_107525*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_cartoon_hidden3_layer_call_and_return_conditional_losses_1071042)
'cartoon_hidden3/StatefulPartitionedCall?
(watermark_output/StatefulPartitionedCallStatefulPartitionedCall2watermark_hidden4/StatefulPartitionedCall:output:0watermark_output_107528watermark_output_107530*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_watermark_output_layer_call_and_return_conditional_losses_1071212*
(watermark_output/StatefulPartitionedCall?
&cartoon_output/StatefulPartitionedCallStatefulPartitionedCall0cartoon_hidden3/StatefulPartitionedCall:output:0cartoon_output_107533cartoon_output_107535*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_cartoon_output_layer_call_and_return_conditional_losses_1071382(
&cartoon_output/StatefulPartitionedCall?
IdentityIdentity/cartoon_output/StatefulPartitionedCall:output:0(^cartoon_hidden1/StatefulPartitionedCall(^cartoon_hidden2/StatefulPartitionedCall(^cartoon_hidden3/StatefulPartitionedCall'^cartoon_output/StatefulPartitionedCall^vgg16/StatefulPartitionedCall*^watermark_hidden1/StatefulPartitionedCall*^watermark_hidden2/StatefulPartitionedCall*^watermark_hidden3/StatefulPartitionedCall*^watermark_hidden4/StatefulPartitionedCall)^watermark_output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity1watermark_output/StatefulPartitionedCall:output:0(^cartoon_hidden1/StatefulPartitionedCall(^cartoon_hidden2/StatefulPartitionedCall(^cartoon_hidden3/StatefulPartitionedCall'^cartoon_output/StatefulPartitionedCall^vgg16/StatefulPartitionedCall*^watermark_hidden1/StatefulPartitionedCall*^watermark_hidden2/StatefulPartitionedCall*^watermark_hidden3/StatefulPartitionedCall*^watermark_hidden4/StatefulPartitionedCall)^watermark_output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesw
u:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2R
'cartoon_hidden1/StatefulPartitionedCall'cartoon_hidden1/StatefulPartitionedCall2R
'cartoon_hidden2/StatefulPartitionedCall'cartoon_hidden2/StatefulPartitionedCall2R
'cartoon_hidden3/StatefulPartitionedCall'cartoon_hidden3/StatefulPartitionedCall2P
&cartoon_output/StatefulPartitionedCall&cartoon_output/StatefulPartitionedCall2>
vgg16/StatefulPartitionedCallvgg16/StatefulPartitionedCall2V
)watermark_hidden1/StatefulPartitionedCall)watermark_hidden1/StatefulPartitionedCall2V
)watermark_hidden2/StatefulPartitionedCall)watermark_hidden2/StatefulPartitionedCall2V
)watermark_hidden3/StatefulPartitionedCall)watermark_hidden3/StatefulPartitionedCall2V
)watermark_hidden4/StatefulPartitionedCall)watermark_hidden4/StatefulPartitionedCall2T
(watermark_output/StatefulPartitionedCall(watermark_output/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
H__inference_block2_conv1_layer_call_and_return_conditional_losses_109136

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2	
BiasAddc
ReluReluBiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?

?
M__inference_watermark_hidden3_layer_call_and_return_conditional_losses_107053

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
L__inference_watermark_output_layer_call_and_return_conditional_losses_107121

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
H__inference_block5_conv1_layer_call_and_return_conditional_losses_106299

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
-__inference_block5_conv1_layer_call_fn_109285

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block5_conv1_layer_call_and_return_conditional_losses_1062992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_vgg16_layer_call_fn_106396
input_1!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_vgg16_layer_call_and_return_conditional_losses_1063412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
b
F__inference_flatten_28_layer_call_and_return_conditional_losses_108896

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ?  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????		?:X T
0
_output_shapes
:?????????		?
 
_user_specified_nameinputs
?
?
-__inference_block3_conv1_layer_call_fn_109165

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????KK?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block3_conv1_layer_call_and_return_conditional_losses_1061952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????KK?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????KK?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????KK?
 
_user_specified_nameinputs
?
?
H__inference_block5_conv3_layer_call_and_return_conditional_losses_109336

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
2__inference_watermark_hidden2_layer_call_fn_108945

inputs
unknown:@ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_watermark_hidden2_layer_call_and_return_conditional_losses_1070192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
0__inference_cartoon_hidden1_layer_call_fn_108925

inputs
unknown:
??@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_cartoon_hidden1_layer_call_and_return_conditional_losses_1070362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
H__inference_block2_conv2_layer_call_and_return_conditional_losses_109156

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2	
BiasAddc
ReluReluBiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
G
input_31;
serving_default_input_31:0???????????B
cartoon_output0
StatefulPartitionedCall:0?????????D
watermark_output0
StatefulPartitionedCall:1?????????tensorflow/serving/predict:??	
??
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer_with_weights-9
layer-12
	optimizer
loss
regularization_losses
trainable_variables
	variables
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"??
_tf_keras_network۝{"name": "model_28", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_28", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 300, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_31"}, "name": "input_31", "inbound_nodes": []}, {"class_name": "Rescaling", "config": {"name": "rescaling_24", "trainable": true, "dtype": "float32", "scale": 0.00392156862745098, "offset": 0.0}, "name": "rescaling_24", "inbound_nodes": [[["input_31", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "vgg16", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 300, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "block1_conv1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1_conv2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv2", "inbound_nodes": [[["block1_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block1_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block1_pool", "inbound_nodes": [[["block1_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv1", "inbound_nodes": [[["block1_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv2", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv2", "inbound_nodes": [[["block2_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block2_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block2_pool", "inbound_nodes": [[["block2_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv1", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv1", "inbound_nodes": [[["block2_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv2", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv2", "inbound_nodes": [[["block3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv3", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv3", "inbound_nodes": [[["block3_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block3_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block3_pool", "inbound_nodes": [[["block3_conv3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv1", "inbound_nodes": [[["block3_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv2", "inbound_nodes": [[["block4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv3", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv3", "inbound_nodes": [[["block4_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block4_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block4_pool", "inbound_nodes": [[["block4_conv3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv1", "inbound_nodes": [[["block4_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv2", "inbound_nodes": [[["block5_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv3", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv3", "inbound_nodes": [[["block5_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block5_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block5_pool", "inbound_nodes": [[["block5_conv3", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["block5_pool", 0, 0]]}, "name": "vgg16", "inbound_nodes": [[["rescaling_24", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_28", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_28", "inbound_nodes": [[["vgg16", 1, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "watermark_hidden1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "watermark_hidden1", "inbound_nodes": [[["flatten_28", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "cartoon_hidden1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "cartoon_hidden1", "inbound_nodes": [[["flatten_28", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "watermark_hidden2", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "watermark_hidden2", "inbound_nodes": [[["watermark_hidden1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "cartoon_hidden2", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "cartoon_hidden2", "inbound_nodes": [[["cartoon_hidden1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "watermark_hidden3", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "watermark_hidden3", "inbound_nodes": [[["watermark_hidden2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "cartoon_hidden3", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "cartoon_hidden3", "inbound_nodes": [[["cartoon_hidden2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "watermark_hidden4", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "watermark_hidden4", "inbound_nodes": [[["watermark_hidden3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "cartoon_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "cartoon_output", "inbound_nodes": [[["cartoon_hidden3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "watermark_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "watermark_output", "inbound_nodes": [[["watermark_hidden4", 0, 0, {}]]]}], "input_layers": [["input_31", 0, 0]], "output_layers": [["cartoon_output", 0, 0], ["watermark_output", 0, 0]]}, "shared_object_id": 76, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 300, 300, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 300, 300, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 300, 300, 3]}, "float32", "input_31"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_28", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 300, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_31"}, "name": "input_31", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Rescaling", "config": {"name": "rescaling_24", "trainable": true, "dtype": "float32", "scale": 0.00392156862745098, "offset": 0.0}, "name": "rescaling_24", "inbound_nodes": [[["input_31", 0, 0, {}]]], "shared_object_id": 1}, {"class_name": "Functional", "config": {"name": "vgg16", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 300, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "block1_conv1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1_conv2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv2", "inbound_nodes": [[["block1_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block1_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block1_pool", "inbound_nodes": [[["block1_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv1", "inbound_nodes": [[["block1_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv2", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv2", "inbound_nodes": [[["block2_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block2_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block2_pool", "inbound_nodes": [[["block2_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv1", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv1", "inbound_nodes": [[["block2_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv2", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv2", "inbound_nodes": [[["block3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv3", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv3", "inbound_nodes": [[["block3_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block3_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block3_pool", "inbound_nodes": [[["block3_conv3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv1", "inbound_nodes": [[["block3_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv2", "inbound_nodes": [[["block4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv3", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv3", "inbound_nodes": [[["block4_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block4_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block4_pool", "inbound_nodes": [[["block4_conv3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv1", "inbound_nodes": [[["block4_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv2", "inbound_nodes": [[["block5_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv3", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv3", "inbound_nodes": [[["block5_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block5_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block5_pool", "inbound_nodes": [[["block5_conv3", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["block5_pool", 0, 0]]}, "name": "vgg16", "inbound_nodes": [[["rescaling_24", 0, 0, {}]]], "shared_object_id": 47}, {"class_name": "Flatten", "config": {"name": "flatten_28", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_28", "inbound_nodes": [[["vgg16", 1, 0, {}]]], "shared_object_id": 48}, {"class_name": "Dense", "config": {"name": "watermark_hidden1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 49}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 50}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "watermark_hidden1", "inbound_nodes": [[["flatten_28", 0, 0, {}]]], "shared_object_id": 51}, {"class_name": "Dense", "config": {"name": "cartoon_hidden1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 52}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 53}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "cartoon_hidden1", "inbound_nodes": [[["flatten_28", 0, 0, {}]]], "shared_object_id": 54}, {"class_name": "Dense", "config": {"name": "watermark_hidden2", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 55}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 56}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "watermark_hidden2", "inbound_nodes": [[["watermark_hidden1", 0, 0, {}]]], "shared_object_id": 57}, {"class_name": "Dense", "config": {"name": "cartoon_hidden2", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 58}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 59}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "cartoon_hidden2", "inbound_nodes": [[["cartoon_hidden1", 0, 0, {}]]], "shared_object_id": 60}, {"class_name": "Dense", "config": {"name": "watermark_hidden3", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 61}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 62}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "watermark_hidden3", "inbound_nodes": [[["watermark_hidden2", 0, 0, {}]]], "shared_object_id": 63}, {"class_name": "Dense", "config": {"name": "cartoon_hidden3", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 64}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 65}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "cartoon_hidden3", "inbound_nodes": [[["cartoon_hidden2", 0, 0, {}]]], "shared_object_id": 66}, {"class_name": "Dense", "config": {"name": "watermark_hidden4", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 67}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 68}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "watermark_hidden4", "inbound_nodes": [[["watermark_hidden3", 0, 0, {}]]], "shared_object_id": 69}, {"class_name": "Dense", "config": {"name": "cartoon_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 70}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 71}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "cartoon_output", "inbound_nodes": [[["cartoon_hidden3", 0, 0, {}]]], "shared_object_id": 72}, {"class_name": "Dense", "config": {"name": "watermark_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 73}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 74}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "watermark_output", "inbound_nodes": [[["watermark_hidden4", 0, 0, {}]]], "shared_object_id": 75}], "input_layers": [["input_31", 0, 0]], "output_layers": [["cartoon_output", 0, 0], ["watermark_output", 0, 0]]}}, "training_config": {"loss": ["binary_crossentropy", null], "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_31", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 300, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 300, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_31"}}
?
	keras_api"?
_tf_keras_layer?{"name": "rescaling_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "Rescaling", "config": {"name": "rescaling_24", "trainable": true, "dtype": "float32", "scale": 0.00392156862745098, "offset": 0.0}, "inbound_nodes": [[["input_31", 0, 0, {}]]], "shared_object_id": 1}
??
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
layer_with_weights-5
layer-8
layer_with_weights-6
layer-9
 layer-10
!layer_with_weights-7
!layer-11
"layer_with_weights-8
"layer-12
#layer_with_weights-9
#layer-13
$layer-14
%layer_with_weights-10
%layer-15
&layer_with_weights-11
&layer-16
'layer_with_weights-12
'layer-17
(layer-18
)regularization_losses
*trainable_variables
+	variables
,	keras_api
?__call__
+?&call_and_return_all_conditional_losses"??
_tf_keras_network??{"name": "vgg16", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "vgg16", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 300, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "block1_conv1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1_conv2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv2", "inbound_nodes": [[["block1_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block1_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block1_pool", "inbound_nodes": [[["block1_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv1", "inbound_nodes": [[["block1_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv2", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv2", "inbound_nodes": [[["block2_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block2_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block2_pool", "inbound_nodes": [[["block2_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv1", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv1", "inbound_nodes": [[["block2_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv2", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv2", "inbound_nodes": [[["block3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv3", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv3", "inbound_nodes": [[["block3_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block3_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block3_pool", "inbound_nodes": [[["block3_conv3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv1", "inbound_nodes": [[["block3_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv2", "inbound_nodes": [[["block4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv3", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv3", "inbound_nodes": [[["block4_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block4_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block4_pool", "inbound_nodes": [[["block4_conv3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv1", "inbound_nodes": [[["block4_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv2", "inbound_nodes": [[["block5_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv3", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv3", "inbound_nodes": [[["block5_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block5_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block5_pool", "inbound_nodes": [[["block5_conv3", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["block5_pool", 0, 0]]}, "inbound_nodes": [[["rescaling_24", 0, 0, {}]]], "shared_object_id": 47, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 300, 300, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 300, 300, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 300, 300, 3]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "vgg16", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 300, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 2}, {"class_name": "Conv2D", "config": {"name": "block1_conv1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 3}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv1", "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "Conv2D", "config": {"name": "block1_conv2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv2", "inbound_nodes": [[["block1_conv1", 0, 0, {}]]], "shared_object_id": 8}, {"class_name": "MaxPooling2D", "config": {"name": "block1_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block1_pool", "inbound_nodes": [[["block1_conv2", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Conv2D", "config": {"name": "block2_conv1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv1", "inbound_nodes": [[["block1_pool", 0, 0, {}]]], "shared_object_id": 12}, {"class_name": "Conv2D", "config": {"name": "block2_conv2", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv2", "inbound_nodes": [[["block2_conv1", 0, 0, {}]]], "shared_object_id": 15}, {"class_name": "MaxPooling2D", "config": {"name": "block2_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block2_pool", "inbound_nodes": [[["block2_conv2", 0, 0, {}]]], "shared_object_id": 16}, {"class_name": "Conv2D", "config": {"name": "block3_conv1", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 17}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv1", "inbound_nodes": [[["block2_pool", 0, 0, {}]]], "shared_object_id": 19}, {"class_name": "Conv2D", "config": {"name": "block3_conv2", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv2", "inbound_nodes": [[["block3_conv1", 0, 0, {}]]], "shared_object_id": 22}, {"class_name": "Conv2D", "config": {"name": "block3_conv3", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 23}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv3", "inbound_nodes": [[["block3_conv2", 0, 0, {}]]], "shared_object_id": 25}, {"class_name": "MaxPooling2D", "config": {"name": "block3_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block3_pool", "inbound_nodes": [[["block3_conv3", 0, 0, {}]]], "shared_object_id": 26}, {"class_name": "Conv2D", "config": {"name": "block4_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 27}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 28}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv1", "inbound_nodes": [[["block3_pool", 0, 0, {}]]], "shared_object_id": 29}, {"class_name": "Conv2D", "config": {"name": "block4_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 30}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 31}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv2", "inbound_nodes": [[["block4_conv1", 0, 0, {}]]], "shared_object_id": 32}, {"class_name": "Conv2D", "config": {"name": "block4_conv3", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 33}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 34}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv3", "inbound_nodes": [[["block4_conv2", 0, 0, {}]]], "shared_object_id": 35}, {"class_name": "MaxPooling2D", "config": {"name": "block4_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block4_pool", "inbound_nodes": [[["block4_conv3", 0, 0, {}]]], "shared_object_id": 36}, {"class_name": "Conv2D", "config": {"name": "block5_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 37}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 38}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv1", "inbound_nodes": [[["block4_pool", 0, 0, {}]]], "shared_object_id": 39}, {"class_name": "Conv2D", "config": {"name": "block5_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 40}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 41}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv2", "inbound_nodes": [[["block5_conv1", 0, 0, {}]]], "shared_object_id": 42}, {"class_name": "Conv2D", "config": {"name": "block5_conv3", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 43}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 44}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv3", "inbound_nodes": [[["block5_conv2", 0, 0, {}]]], "shared_object_id": 45}, {"class_name": "MaxPooling2D", "config": {"name": "block5_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block5_pool", "inbound_nodes": [[["block5_conv3", 0, 0, {}]]], "shared_object_id": 46}], "input_layers": [["input_1", 0, 0]], "output_layers": [["block5_pool", 0, 0]]}}}
?
-regularization_losses
.trainable_variables
/	variables
0	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "flatten_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_28", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["vgg16", 1, 0, {}]]], "shared_object_id": 48, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 79}}
?	

1kernel
2bias
3regularization_losses
4trainable_variables
5	variables
6	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "watermark_hidden1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "watermark_hidden1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 49}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 50}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["flatten_28", 0, 0, {}]]], "shared_object_id": 51, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 41472}}, "shared_object_id": 80}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 41472]}}
?	

7kernel
8bias
9regularization_losses
:trainable_variables
;	variables
<	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "cartoon_hidden1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "cartoon_hidden1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 52}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 53}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["flatten_28", 0, 0, {}]]], "shared_object_id": 54, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 41472}}, "shared_object_id": 81}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 41472]}}
?	

=kernel
>bias
?regularization_losses
@trainable_variables
A	variables
B	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "watermark_hidden2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "watermark_hidden2", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 55}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 56}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["watermark_hidden1", 0, 0, {}]]], "shared_object_id": 57, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 82}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?	

Ckernel
Dbias
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "cartoon_hidden2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "cartoon_hidden2", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 58}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 59}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["cartoon_hidden1", 0, 0, {}]]], "shared_object_id": 60, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 83}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?	

Ikernel
Jbias
Kregularization_losses
Ltrainable_variables
M	variables
N	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "watermark_hidden3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "watermark_hidden3", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 61}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 62}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["watermark_hidden2", 0, 0, {}]]], "shared_object_id": 63, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 84}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?	

Okernel
Pbias
Qregularization_losses
Rtrainable_variables
S	variables
T	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "cartoon_hidden3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "cartoon_hidden3", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 64}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 65}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["cartoon_hidden2", 0, 0, {}]]], "shared_object_id": 66, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}, "shared_object_id": 85}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
?	

Ukernel
Vbias
Wregularization_losses
Xtrainable_variables
Y	variables
Z	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "watermark_hidden4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "watermark_hidden4", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 67}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 68}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["watermark_hidden3", 0, 0, {}]]], "shared_object_id": 69, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}, "shared_object_id": 86}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
?	

[kernel
\bias
]regularization_losses
^trainable_variables
_	variables
`	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "cartoon_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "cartoon_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 70}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 71}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["cartoon_hidden3", 0, 0, {}]]], "shared_object_id": 72, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}, "shared_object_id": 87}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
?	

akernel
bbias
cregularization_losses
dtrainable_variables
e	variables
f	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "watermark_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "watermark_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 73}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 74}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["watermark_hidden4", 0, 0, {}]]], "shared_object_id": 75, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}, "shared_object_id": 88}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
"
	optimizer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
10
21
72
83
=4
>5
C6
D7
I8
J9
O10
P11
U12
V13
[14
\15
a16
b17"
trackable_list_wrapper
?
g0
h1
i2
j3
k4
l5
m6
n7
o8
p9
q10
r11
s12
t13
u14
v15
w16
x17
y18
z19
{20
|21
}22
~23
24
?25
126
227
728
829
=30
>31
C32
D33
I34
J35
O36
P37
U38
V39
[40
\41
a42
b43"
trackable_list_wrapper
?
regularization_losses
?layer_metrics
trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
	variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
"
_generic_user_object
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 300, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 300, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?

gkernel
hbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "block1_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "block1_conv1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 3}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}, "shared_object_id": 89}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 300, 300, 3]}}
?

ikernel
jbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "block1_conv2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "block1_conv2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block1_conv1", 0, 0, {}]]], "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 90}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 300, 300, 64]}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "block1_pool", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "block1_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["block1_conv2", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 91}}
?

kkernel
lbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "block2_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "block2_conv1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block1_pool", 0, 0, {}]]], "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 92}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 150, 150, 64]}}
?

mkernel
nbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "block2_conv2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "block2_conv2", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block2_conv1", 0, 0, {}]]], "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 93}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 150, 150, 128]}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "block2_pool", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "block2_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["block2_conv2", 0, 0, {}]]], "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 94}}
?

okernel
pbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "block3_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "block3_conv1", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 17}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block2_pool", 0, 0, {}]]], "shared_object_id": 19, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 95}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 75, 75, 128]}}
?

qkernel
rbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "block3_conv2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "block3_conv2", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block3_conv1", 0, 0, {}]]], "shared_object_id": 22, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}, "shared_object_id": 96}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 75, 75, 256]}}
?

skernel
tbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "block3_conv3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "block3_conv3", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 23}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block3_conv2", 0, 0, {}]]], "shared_object_id": 25, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}, "shared_object_id": 97}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 75, 75, 256]}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "block3_pool", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "block3_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["block3_conv3", 0, 0, {}]]], "shared_object_id": 26, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 98}}
?

ukernel
vbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "block4_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "block4_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 27}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 28}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block3_pool", 0, 0, {}]]], "shared_object_id": 29, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}, "shared_object_id": 99}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 37, 37, 256]}}
?

wkernel
xbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "block4_conv2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "block4_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 30}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 31}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block4_conv1", 0, 0, {}]]], "shared_object_id": 32, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 512}}, "shared_object_id": 100}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 37, 37, 512]}}
?

ykernel
zbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "block4_conv3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "block4_conv3", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 33}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 34}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block4_conv2", 0, 0, {}]]], "shared_object_id": 35, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 512}}, "shared_object_id": 101}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 37, 37, 512]}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "block4_pool", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "block4_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["block4_conv3", 0, 0, {}]]], "shared_object_id": 36, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 102}}
?

{kernel
|bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "block5_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "block5_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 37}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 38}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block4_pool", 0, 0, {}]]], "shared_object_id": 39, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 512}}, "shared_object_id": 103}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 18, 18, 512]}}
?

}kernel
~bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "block5_conv2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "block5_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 40}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 41}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block5_conv1", 0, 0, {}]]], "shared_object_id": 42, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 512}}, "shared_object_id": 104}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 18, 18, 512]}}
?

kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "block5_conv3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "block5_conv3", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 43}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 44}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block5_conv2", 0, 0, {}]]], "shared_object_id": 45, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 512}}, "shared_object_id": 105}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 18, 18, 512]}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "block5_pool", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "block5_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["block5_conv3", 0, 0, {}]]], "shared_object_id": 46, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 106}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
g0
h1
i2
j3
k4
l5
m6
n7
o8
p9
q10
r11
s12
t13
u14
v15
w16
x17
y18
z19
{20
|21
}22
~23
24
?25"
trackable_list_wrapper
?
)regularization_losses
?layer_metrics
*trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
+	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
-regularization_losses
?layer_metrics
.trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
/	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*
??@2watermark_hidden1/kernel
$:"@2watermark_hidden1/bias
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
?
3regularization_losses
?layer_metrics
4trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
5	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(
??@2cartoon_hidden1/kernel
": @2cartoon_hidden1/bias
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
?
9regularization_losses
?layer_metrics
:trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
;	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(@ 2watermark_hidden2/kernel
$:" 2watermark_hidden2/bias
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
@trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
A	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:&@2cartoon_hidden2/kernel
": 2cartoon_hidden2/bias
 "
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
?
Eregularization_losses
?layer_metrics
Ftrainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
G	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( 2watermark_hidden3/kernel
$:"2watermark_hidden3/bias
 "
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
?
Kregularization_losses
?layer_metrics
Ltrainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
M	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:&
2cartoon_hidden3/kernel
": 
2cartoon_hidden3/bias
 "
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
?
Qregularization_losses
?layer_metrics
Rtrainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
S	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(
2watermark_hidden4/kernel
$:"
2watermark_hidden4/bias
 "
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
?
Wregularization_losses
?layer_metrics
Xtrainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
Y	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%
2cartoon_output/kernel
!:2cartoon_output/bias
 "
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
?
]regularization_losses
?layer_metrics
^trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
_	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'
2watermark_output/kernel
#:!2watermark_output/bias
 "
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
?
cregularization_losses
?layer_metrics
dtrainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
e	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-:+@2block1_conv1/kernel
:@2block1_conv1/bias
-:+@@2block1_conv2/kernel
:@2block1_conv2/bias
.:,@?2block2_conv1/kernel
 :?2block2_conv1/bias
/:-??2block2_conv2/kernel
 :?2block2_conv2/bias
/:-??2block3_conv1/kernel
 :?2block3_conv1/bias
/:-??2block3_conv2/kernel
 :?2block3_conv2/bias
/:-??2block3_conv3/kernel
 :?2block3_conv3/bias
/:-??2block4_conv1/kernel
 :?2block4_conv1/bias
/:-??2block4_conv2/kernel
 :?2block4_conv2/bias
/:-??2block4_conv3/kernel
 :?2block4_conv3/bias
/:-??2block5_conv1/kernel
 :?2block5_conv1/bias
/:-??2block5_conv2/kernel
 :?2block5_conv2/bias
/:-??2block5_conv3/kernel
 :?2block5_conv3/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?
g0
h1
i2
j3
k4
l5
m6
n7
o8
p9
q10
r11
s12
t13
u14
v15
w16
x17
y18
z19
{20
|21
}22
~23
24
?25"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
	8

9
10
11
12"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
/
0
?1"
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layers
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?
g0
h1
i2
j3
k4
l5
m6
n7
o8
p9
q10
r11
s12
t13
u14
v15
w16
x17
y18
z19
{20
|21
}22
~23
24
?25"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
8
9
 10
!11
"12
#13
$14
%15
&16
'17
(18"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 107}
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "cartoon_output_loss", "dtype": "float32", "config": {"name": "cartoon_output_loss", "dtype": "float32"}, "shared_object_id": 108}
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
/
0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
?2?
)__inference_model_28_layer_call_fn_107239
)__inference_model_28_layer_call_fn_108136
)__inference_model_28_layer_call_fn_108231
)__inference_model_28_layer_call_fn_107728?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
!__inference__wrapped_model_106047?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *1?.
,?)
input_31???????????
?2?
D__inference_model_28_layer_call_and_return_conditional_losses_108401
D__inference_model_28_layer_call_and_return_conditional_losses_108571
D__inference_model_28_layer_call_and_return_conditional_losses_107836
D__inference_model_28_layer_call_and_return_conditional_losses_107944?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_vgg16_layer_call_fn_106396
&__inference_vgg16_layer_call_fn_108628
&__inference_vgg16_layer_call_fn_108685
&__inference_vgg16_layer_call_fn_106771?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_vgg16_layer_call_and_return_conditional_losses_108785
A__inference_vgg16_layer_call_and_return_conditional_losses_108885
A__inference_vgg16_layer_call_and_return_conditional_losses_106845
A__inference_vgg16_layer_call_and_return_conditional_losses_106919?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_flatten_28_layer_call_fn_108890?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_flatten_28_layer_call_and_return_conditional_losses_108896?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_watermark_hidden1_layer_call_fn_108905?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_watermark_hidden1_layer_call_and_return_conditional_losses_108916?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_cartoon_hidden1_layer_call_fn_108925?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_cartoon_hidden1_layer_call_and_return_conditional_losses_108936?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_watermark_hidden2_layer_call_fn_108945?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_watermark_hidden2_layer_call_and_return_conditional_losses_108956?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_cartoon_hidden2_layer_call_fn_108965?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_cartoon_hidden2_layer_call_and_return_conditional_losses_108976?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_watermark_hidden3_layer_call_fn_108985?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_watermark_hidden3_layer_call_and_return_conditional_losses_108996?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_cartoon_hidden3_layer_call_fn_109005?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_cartoon_hidden3_layer_call_and_return_conditional_losses_109016?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_watermark_hidden4_layer_call_fn_109025?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_watermark_hidden4_layer_call_and_return_conditional_losses_109036?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_cartoon_output_layer_call_fn_109045?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_cartoon_output_layer_call_and_return_conditional_losses_109056?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_watermark_output_layer_call_fn_109065?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_watermark_output_layer_call_and_return_conditional_losses_109076?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_108041input_31"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_block1_conv1_layer_call_fn_109085?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_block1_conv1_layer_call_and_return_conditional_losses_109096?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_block1_conv2_layer_call_fn_109105?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_block1_conv2_layer_call_and_return_conditional_losses_109116?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_block1_pool_layer_call_fn_106059?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
G__inference_block1_pool_layer_call_and_return_conditional_losses_106053?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
-__inference_block2_conv1_layer_call_fn_109125?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_block2_conv1_layer_call_and_return_conditional_losses_109136?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_block2_conv2_layer_call_fn_109145?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_block2_conv2_layer_call_and_return_conditional_losses_109156?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_block2_pool_layer_call_fn_106071?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
G__inference_block2_pool_layer_call_and_return_conditional_losses_106065?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
-__inference_block3_conv1_layer_call_fn_109165?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_block3_conv1_layer_call_and_return_conditional_losses_109176?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_block3_conv2_layer_call_fn_109185?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_block3_conv2_layer_call_and_return_conditional_losses_109196?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_block3_conv3_layer_call_fn_109205?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_block3_conv3_layer_call_and_return_conditional_losses_109216?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_block3_pool_layer_call_fn_106083?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
G__inference_block3_pool_layer_call_and_return_conditional_losses_106077?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
-__inference_block4_conv1_layer_call_fn_109225?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_block4_conv1_layer_call_and_return_conditional_losses_109236?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_block4_conv2_layer_call_fn_109245?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_block4_conv2_layer_call_and_return_conditional_losses_109256?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_block4_conv3_layer_call_fn_109265?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_block4_conv3_layer_call_and_return_conditional_losses_109276?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_block4_pool_layer_call_fn_106095?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
G__inference_block4_pool_layer_call_and_return_conditional_losses_106089?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
-__inference_block5_conv1_layer_call_fn_109285?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_block5_conv1_layer_call_and_return_conditional_losses_109296?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_block5_conv2_layer_call_fn_109305?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_block5_conv2_layer_call_and_return_conditional_losses_109316?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_block5_conv3_layer_call_fn_109325?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_block5_conv3_layer_call_and_return_conditional_losses_109336?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_block5_pool_layer_call_fn_106107?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
G__inference_block5_pool_layer_call_and_return_conditional_losses_106101?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84?????????????????????????????????????
!__inference__wrapped_model_106047?-ghijklmnopqrstuvwxyz{|}~?12=>78IJCDUVOPab[\;?8
1?.
,?)
input_31???????????
? "?|
:
cartoon_output(?%
cartoon_output?????????
>
watermark_output*?'
watermark_output??????????
H__inference_block1_conv1_layer_call_and_return_conditional_losses_109096pgh9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????@
? ?
-__inference_block1_conv1_layer_call_fn_109085cgh9?6
/?,
*?'
inputs???????????
? ""????????????@?
H__inference_block1_conv2_layer_call_and_return_conditional_losses_109116pij9?6
/?,
*?'
inputs???????????@
? "/?,
%?"
0???????????@
? ?
-__inference_block1_conv2_layer_call_fn_109105cij9?6
/?,
*?'
inputs???????????@
? ""????????????@?
G__inference_block1_pool_layer_call_and_return_conditional_losses_106053?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
,__inference_block1_pool_layer_call_fn_106059?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
H__inference_block2_conv1_layer_call_and_return_conditional_losses_109136qkl9?6
/?,
*?'
inputs???????????@
? "0?-
&?#
0????????????
? ?
-__inference_block2_conv1_layer_call_fn_109125dkl9?6
/?,
*?'
inputs???????????@
? "#? ?????????????
H__inference_block2_conv2_layer_call_and_return_conditional_losses_109156rmn:?7
0?-
+?(
inputs????????????
? "0?-
&?#
0????????????
? ?
-__inference_block2_conv2_layer_call_fn_109145emn:?7
0?-
+?(
inputs????????????
? "#? ?????????????
G__inference_block2_pool_layer_call_and_return_conditional_losses_106065?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
,__inference_block2_pool_layer_call_fn_106071?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
H__inference_block3_conv1_layer_call_and_return_conditional_losses_109176nop8?5
.?+
)?&
inputs?????????KK?
? ".?+
$?!
0?????????KK?
? ?
-__inference_block3_conv1_layer_call_fn_109165aop8?5
.?+
)?&
inputs?????????KK?
? "!??????????KK??
H__inference_block3_conv2_layer_call_and_return_conditional_losses_109196nqr8?5
.?+
)?&
inputs?????????KK?
? ".?+
$?!
0?????????KK?
? ?
-__inference_block3_conv2_layer_call_fn_109185aqr8?5
.?+
)?&
inputs?????????KK?
? "!??????????KK??
H__inference_block3_conv3_layer_call_and_return_conditional_losses_109216nst8?5
.?+
)?&
inputs?????????KK?
? ".?+
$?!
0?????????KK?
? ?
-__inference_block3_conv3_layer_call_fn_109205ast8?5
.?+
)?&
inputs?????????KK?
? "!??????????KK??
G__inference_block3_pool_layer_call_and_return_conditional_losses_106077?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
,__inference_block3_pool_layer_call_fn_106083?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
H__inference_block4_conv1_layer_call_and_return_conditional_losses_109236nuv8?5
.?+
)?&
inputs?????????%%?
? ".?+
$?!
0?????????%%?
? ?
-__inference_block4_conv1_layer_call_fn_109225auv8?5
.?+
)?&
inputs?????????%%?
? "!??????????%%??
H__inference_block4_conv2_layer_call_and_return_conditional_losses_109256nwx8?5
.?+
)?&
inputs?????????%%?
? ".?+
$?!
0?????????%%?
? ?
-__inference_block4_conv2_layer_call_fn_109245awx8?5
.?+
)?&
inputs?????????%%?
? "!??????????%%??
H__inference_block4_conv3_layer_call_and_return_conditional_losses_109276nyz8?5
.?+
)?&
inputs?????????%%?
? ".?+
$?!
0?????????%%?
? ?
-__inference_block4_conv3_layer_call_fn_109265ayz8?5
.?+
)?&
inputs?????????%%?
? "!??????????%%??
G__inference_block4_pool_layer_call_and_return_conditional_losses_106089?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
,__inference_block4_pool_layer_call_fn_106095?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
H__inference_block5_conv1_layer_call_and_return_conditional_losses_109296n{|8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
-__inference_block5_conv1_layer_call_fn_109285a{|8?5
.?+
)?&
inputs??????????
? "!????????????
H__inference_block5_conv2_layer_call_and_return_conditional_losses_109316n}~8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
-__inference_block5_conv2_layer_call_fn_109305a}~8?5
.?+
)?&
inputs??????????
? "!????????????
H__inference_block5_conv3_layer_call_and_return_conditional_losses_109336o?8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
-__inference_block5_conv3_layer_call_fn_109325b?8?5
.?+
)?&
inputs??????????
? "!????????????
G__inference_block5_pool_layer_call_and_return_conditional_losses_106101?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
,__inference_block5_pool_layer_call_fn_106107?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
K__inference_cartoon_hidden1_layer_call_and_return_conditional_losses_108936^781?.
'?$
"?
inputs???????????
? "%?"
?
0?????????@
? ?
0__inference_cartoon_hidden1_layer_call_fn_108925Q781?.
'?$
"?
inputs???????????
? "??????????@?
K__inference_cartoon_hidden2_layer_call_and_return_conditional_losses_108976\CD/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? ?
0__inference_cartoon_hidden2_layer_call_fn_108965OCD/?,
%?"
 ?
inputs?????????@
? "???????????
K__inference_cartoon_hidden3_layer_call_and_return_conditional_losses_109016\OP/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????

? ?
0__inference_cartoon_hidden3_layer_call_fn_109005OOP/?,
%?"
 ?
inputs?????????
? "??????????
?
J__inference_cartoon_output_layer_call_and_return_conditional_losses_109056\[\/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????
? ?
/__inference_cartoon_output_layer_call_fn_109045O[\/?,
%?"
 ?
inputs?????????

? "???????????
F__inference_flatten_28_layer_call_and_return_conditional_losses_108896c8?5
.?+
)?&
inputs?????????		?
? "'?$
?
0???????????
? ?
+__inference_flatten_28_layer_call_fn_108890V8?5
.?+
)?&
inputs?????????		?
? "?????????????
D__inference_model_28_layer_call_and_return_conditional_losses_107836?-ghijklmnopqrstuvwxyz{|}~?12=>78IJCDUVOPab[\C?@
9?6
,?)
input_31???????????
p 

 
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
D__inference_model_28_layer_call_and_return_conditional_losses_107944?-ghijklmnopqrstuvwxyz{|}~?12=>78IJCDUVOPab[\C?@
9?6
,?)
input_31???????????
p

 
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
D__inference_model_28_layer_call_and_return_conditional_losses_108401?-ghijklmnopqrstuvwxyz{|}~?12=>78IJCDUVOPab[\A?>
7?4
*?'
inputs???????????
p 

 
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
D__inference_model_28_layer_call_and_return_conditional_losses_108571?-ghijklmnopqrstuvwxyz{|}~?12=>78IJCDUVOPab[\A?>
7?4
*?'
inputs???????????
p

 
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
)__inference_model_28_layer_call_fn_107239?-ghijklmnopqrstuvwxyz{|}~?12=>78IJCDUVOPab[\C?@
9?6
,?)
input_31???????????
p 

 
? "=?:
?
0?????????
?
1??????????
)__inference_model_28_layer_call_fn_107728?-ghijklmnopqrstuvwxyz{|}~?12=>78IJCDUVOPab[\C?@
9?6
,?)
input_31???????????
p

 
? "=?:
?
0?????????
?
1??????????
)__inference_model_28_layer_call_fn_108136?-ghijklmnopqrstuvwxyz{|}~?12=>78IJCDUVOPab[\A?>
7?4
*?'
inputs???????????
p 

 
? "=?:
?
0?????????
?
1??????????
)__inference_model_28_layer_call_fn_108231?-ghijklmnopqrstuvwxyz{|}~?12=>78IJCDUVOPab[\A?>
7?4
*?'
inputs???????????
p

 
? "=?:
?
0?????????
?
1??????????
$__inference_signature_wrapper_108041?-ghijklmnopqrstuvwxyz{|}~?12=>78IJCDUVOPab[\G?D
? 
=?:
8
input_31,?)
input_31???????????"?|
:
cartoon_output(?%
cartoon_output?????????
>
watermark_output*?'
watermark_output??????????
A__inference_vgg16_layer_call_and_return_conditional_losses_106845?ghijklmnopqrstuvwxyz{|}~?B??
8?5
+?(
input_1???????????
p 

 
? ".?+
$?!
0?????????		?
? ?
A__inference_vgg16_layer_call_and_return_conditional_losses_106919?ghijklmnopqrstuvwxyz{|}~?B??
8?5
+?(
input_1???????????
p

 
? ".?+
$?!
0?????????		?
? ?
A__inference_vgg16_layer_call_and_return_conditional_losses_108785?ghijklmnopqrstuvwxyz{|}~?A?>
7?4
*?'
inputs???????????
p 

 
? ".?+
$?!
0?????????		?
? ?
A__inference_vgg16_layer_call_and_return_conditional_losses_108885?ghijklmnopqrstuvwxyz{|}~?A?>
7?4
*?'
inputs???????????
p

 
? ".?+
$?!
0?????????		?
? ?
&__inference_vgg16_layer_call_fn_106396?ghijklmnopqrstuvwxyz{|}~?B??
8?5
+?(
input_1???????????
p 

 
? "!??????????		??
&__inference_vgg16_layer_call_fn_106771?ghijklmnopqrstuvwxyz{|}~?B??
8?5
+?(
input_1???????????
p

 
? "!??????????		??
&__inference_vgg16_layer_call_fn_108628?ghijklmnopqrstuvwxyz{|}~?A?>
7?4
*?'
inputs???????????
p 

 
? "!??????????		??
&__inference_vgg16_layer_call_fn_108685?ghijklmnopqrstuvwxyz{|}~?A?>
7?4
*?'
inputs???????????
p

 
? "!??????????		??
M__inference_watermark_hidden1_layer_call_and_return_conditional_losses_108916^121?.
'?$
"?
inputs???????????
? "%?"
?
0?????????@
? ?
2__inference_watermark_hidden1_layer_call_fn_108905Q121?.
'?$
"?
inputs???????????
? "??????????@?
M__inference_watermark_hidden2_layer_call_and_return_conditional_losses_108956\=>/?,
%?"
 ?
inputs?????????@
? "%?"
?
0????????? 
? ?
2__inference_watermark_hidden2_layer_call_fn_108945O=>/?,
%?"
 ?
inputs?????????@
? "?????????? ?
M__inference_watermark_hidden3_layer_call_and_return_conditional_losses_108996\IJ/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? ?
2__inference_watermark_hidden3_layer_call_fn_108985OIJ/?,
%?"
 ?
inputs????????? 
? "???????????
M__inference_watermark_hidden4_layer_call_and_return_conditional_losses_109036\UV/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????

? ?
2__inference_watermark_hidden4_layer_call_fn_109025OUV/?,
%?"
 ?
inputs?????????
? "??????????
?
L__inference_watermark_output_layer_call_and_return_conditional_losses_109076\ab/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????
? ?
1__inference_watermark_output_layer_call_fn_109065Oab/?,
%?"
 ?
inputs?????????

? "??????????