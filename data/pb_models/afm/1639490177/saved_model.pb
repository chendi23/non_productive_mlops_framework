??

??
:
Add
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"T?

value"T

output_ref"T?"	
Ttype"
validate_shapebool("
use_lockingbool(?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?
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
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
b
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:

2	
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
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
s

VariableV2
ref"dtype?"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ?"serve*	2.4.0-rc02&tf_macos-v0.1-alpha2-AS-67-gf3595294ab??


global_step/Initializer/zerosConst*
_class
loc:@global_step*
_output_shapes
: *
dtype0	*
value	B	 R 
?
global_step
VariableV2*
_class
loc:@global_step*
_output_shapes
: *
	container *
dtype0	*
shape: *
shared_name 
?
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
T0	*
_class
loc:@global_step*
_output_shapes
: *
use_locking(*
validate_shape(
j
global_step/readIdentityglobal_step*
T0	*
_class
loc:@global_step*
_output_shapes
: 
e
XiPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
e
XvPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
k
inputs/ToInt32CastXi*

DstT0*

SrcT0*
Truncate( *'
_output_shapes
:?????????
o
embeddings/random_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB"O  
   
b
embeddings/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
d
embeddings/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
-embeddings/random_normal/RandomStandardNormalRandomStandardNormalembeddings/random_normal/shape*
T0*
_output_shapes
:	?
*
dtype0*
seed?*
seed2 
?
embeddings/random_normal/mulMul-embeddings/random_normal/RandomStandardNormalembeddings/random_normal/stddev*
T0*
_output_shapes
:	?

?
embeddings/random_normalAddembeddings/random_normal/mulembeddings/random_normal/mean*
T0*
_output_shapes
:	?

?
embeddings/Variable
VariableV2*
_output_shapes
:	?
*
	container *
dtype0*
shape:	?
*
shared_name 
?
embeddings/Variable/AssignAssignembeddings/Variableembeddings/random_normal*
T0*&
_class
loc:@embeddings/Variable*
_output_shapes
:	?
*
use_locking(*
validate_shape(
?
embeddings/Variable/readIdentityembeddings/Variable*
T0*&
_class
loc:@embeddings/Variable*
_output_shapes
:	?

?
 embeddings/embedding_lookup/axisConst*&
_class
loc:@embeddings/Variable*
_output_shapes
: *
dtype0*
value	B : 
?
embeddings/embedding_lookupGatherV2embeddings/Variable/readinputs/ToInt32 embeddings/embedding_lookup/axis*
Taxis0*
Tindices0*
Tparams0*&
_class
loc:@embeddings/Variable*+
_output_shapes
:?????????
*

batch_dims 
?
$embeddings/embedding_lookup/IdentityIdentityembeddings/embedding_lookup*
T0*+
_output_shapes
:?????????

m
embeddings/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      

embeddings/ReshapeReshapeXvembeddings/Reshape/shape*
T0*
Tshape0*+
_output_shapes
:?????????
?
embeddings/embeddings_outMul$embeddings/embedding_lookup/Identityembeddings/Reshape*
T0*+
_output_shapes
:?????????

~
)interactive_attention/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
?
+interactive_attention/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
+interactive_attention/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
#interactive_attention/strided_sliceStridedSliceembeddings/embeddings_out)interactive_attention/strided_slice/stack+interactive_attention/strided_slice/stack_1+interactive_attention/strided_slice/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
+interactive_attention/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
-interactive_attention/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
-interactive_attention/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
%interactive_attention/strided_slice_1StridedSliceembeddings/embeddings_out+interactive_attention/strided_slice_1/stack-interactive_attention/strided_slice_1/stack_1-interactive_attention/strided_slice_1/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/MulMul#interactive_attention/strided_slice%interactive_attention/strided_slice_1*
T0*'
_output_shapes
:?????????

?
+interactive_attention/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
?
-interactive_attention/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
-interactive_attention/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
%interactive_attention/strided_slice_2StridedSliceembeddings/embeddings_out+interactive_attention/strided_slice_2/stack-interactive_attention/strided_slice_2/stack_1-interactive_attention/strided_slice_2/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
+interactive_attention/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
-interactive_attention/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
-interactive_attention/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
%interactive_attention/strided_slice_3StridedSliceembeddings/embeddings_out+interactive_attention/strided_slice_3/stack-interactive_attention/strided_slice_3/stack_1-interactive_attention/strided_slice_3/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_1Mul%interactive_attention/strided_slice_2%interactive_attention/strided_slice_3*
T0*'
_output_shapes
:?????????

?
+interactive_attention/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
?
-interactive_attention/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
-interactive_attention/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
%interactive_attention/strided_slice_4StridedSliceembeddings/embeddings_out+interactive_attention/strided_slice_4/stack-interactive_attention/strided_slice_4/stack_1-interactive_attention/strided_slice_4/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
+interactive_attention/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
-interactive_attention/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
-interactive_attention/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
%interactive_attention/strided_slice_5StridedSliceembeddings/embeddings_out+interactive_attention/strided_slice_5/stack-interactive_attention/strided_slice_5/stack_1-interactive_attention/strided_slice_5/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_2Mul%interactive_attention/strided_slice_4%interactive_attention/strided_slice_5*
T0*'
_output_shapes
:?????????

?
+interactive_attention/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
?
-interactive_attention/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
-interactive_attention/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
%interactive_attention/strided_slice_6StridedSliceembeddings/embeddings_out+interactive_attention/strided_slice_6/stack-interactive_attention/strided_slice_6/stack_1-interactive_attention/strided_slice_6/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
+interactive_attention/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
-interactive_attention/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
-interactive_attention/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
%interactive_attention/strided_slice_7StridedSliceembeddings/embeddings_out+interactive_attention/strided_slice_7/stack-interactive_attention/strided_slice_7/stack_1-interactive_attention/strided_slice_7/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_3Mul%interactive_attention/strided_slice_6%interactive_attention/strided_slice_7*
T0*'
_output_shapes
:?????????

?
+interactive_attention/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
?
-interactive_attention/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
-interactive_attention/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
%interactive_attention/strided_slice_8StridedSliceembeddings/embeddings_out+interactive_attention/strided_slice_8/stack-interactive_attention/strided_slice_8/stack_1-interactive_attention/strided_slice_8/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
+interactive_attention/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
-interactive_attention/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
-interactive_attention/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
%interactive_attention/strided_slice_9StridedSliceembeddings/embeddings_out+interactive_attention/strided_slice_9/stack-interactive_attention/strided_slice_9/stack_1-interactive_attention/strided_slice_9/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_4Mul%interactive_attention/strided_slice_8%interactive_attention/strided_slice_9*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
?
.interactive_attention/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_10StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_10/stack.interactive_attention/strided_slice_10/stack_1.interactive_attention/strided_slice_10/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_11StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_11/stack.interactive_attention/strided_slice_11/stack_1.interactive_attention/strided_slice_11/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_5Mul&interactive_attention/strided_slice_10&interactive_attention/strided_slice_11*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_12/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
?
.interactive_attention/strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_12StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_12/stack.interactive_attention/strided_slice_12/stack_1.interactive_attention/strided_slice_12/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_13/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_13StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_13/stack.interactive_attention/strided_slice_13/stack_1.interactive_attention/strided_slice_13/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_6Mul&interactive_attention/strided_slice_12&interactive_attention/strided_slice_13*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_14/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
?
.interactive_attention/strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_14StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_14/stack.interactive_attention/strided_slice_14/stack_1.interactive_attention/strided_slice_14/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_15/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    	       
?
.interactive_attention/strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_15StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_15/stack.interactive_attention/strided_slice_15/stack_1.interactive_attention/strided_slice_15/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_7Mul&interactive_attention/strided_slice_14&interactive_attention/strided_slice_15*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_16/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
?
.interactive_attention/strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_16StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_16/stack.interactive_attention/strided_slice_16/stack_1.interactive_attention/strided_slice_16/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_17/stackConst*
_output_shapes
:*
dtype0*!
valueB"    	       
?
.interactive_attention/strided_slice_17/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    
       
?
.interactive_attention/strided_slice_17/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_17StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_17/stack.interactive_attention/strided_slice_17/stack_1.interactive_attention/strided_slice_17/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_8Mul&interactive_attention/strided_slice_16&interactive_attention/strided_slice_17*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_18/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
?
.interactive_attention/strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_18StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_18/stack.interactive_attention/strided_slice_18/stack_1.interactive_attention/strided_slice_18/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_19/stackConst*
_output_shapes
:*
dtype0*!
valueB"    
       
?
.interactive_attention/strided_slice_19/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_19/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_19StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_19/stack.interactive_attention/strided_slice_19/stack_1.interactive_attention/strided_slice_19/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_9Mul&interactive_attention/strided_slice_18&interactive_attention/strided_slice_19*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_20/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
?
.interactive_attention/strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_20StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_20/stack.interactive_attention/strided_slice_20/stack_1.interactive_attention/strided_slice_20/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_21/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_21/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_21/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_21StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_21/stack.interactive_attention/strided_slice_21/stack_1.interactive_attention/strided_slice_21/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_10Mul&interactive_attention/strided_slice_20&interactive_attention/strided_slice_21*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_22/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
?
.interactive_attention/strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_22StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_22/stack.interactive_attention/strided_slice_22/stack_1.interactive_attention/strided_slice_22/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_23/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_23/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_23/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_23StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_23/stack.interactive_attention/strided_slice_23/stack_1.interactive_attention/strided_slice_23/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_11Mul&interactive_attention/strided_slice_22&interactive_attention/strided_slice_23*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_24/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_24/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_24/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_24StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_24/stack.interactive_attention/strided_slice_24/stack_1.interactive_attention/strided_slice_24/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_25/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_25/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_25/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_25StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_25/stack.interactive_attention/strided_slice_25/stack_1.interactive_attention/strided_slice_25/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_12Mul&interactive_attention/strided_slice_24&interactive_attention/strided_slice_25*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_26/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_26/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_26/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_26StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_26/stack.interactive_attention/strided_slice_26/stack_1.interactive_attention/strided_slice_26/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_27/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_27/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_27/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_27StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_27/stack.interactive_attention/strided_slice_27/stack_1.interactive_attention/strided_slice_27/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_13Mul&interactive_attention/strided_slice_26&interactive_attention/strided_slice_27*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_28/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_28/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_28/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_28StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_28/stack.interactive_attention/strided_slice_28/stack_1.interactive_attention/strided_slice_28/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_29/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_29/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_29/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_29StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_29/stack.interactive_attention/strided_slice_29/stack_1.interactive_attention/strided_slice_29/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_14Mul&interactive_attention/strided_slice_28&interactive_attention/strided_slice_29*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_30/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_30/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_30/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_30StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_30/stack.interactive_attention/strided_slice_30/stack_1.interactive_attention/strided_slice_30/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_31/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_31/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_31/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_31StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_31/stack.interactive_attention/strided_slice_31/stack_1.interactive_attention/strided_slice_31/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_15Mul&interactive_attention/strided_slice_30&interactive_attention/strided_slice_31*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_32/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_32/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_32/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_32StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_32/stack.interactive_attention/strided_slice_32/stack_1.interactive_attention/strided_slice_32/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_33/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_33/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_33/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_33StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_33/stack.interactive_attention/strided_slice_33/stack_1.interactive_attention/strided_slice_33/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_16Mul&interactive_attention/strided_slice_32&interactive_attention/strided_slice_33*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_34/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_34/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_34/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_34StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_34/stack.interactive_attention/strided_slice_34/stack_1.interactive_attention/strided_slice_34/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_35/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_35/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_35/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_35StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_35/stack.interactive_attention/strided_slice_35/stack_1.interactive_attention/strided_slice_35/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_17Mul&interactive_attention/strided_slice_34&interactive_attention/strided_slice_35*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_36/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_36/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_36/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_36StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_36/stack.interactive_attention/strided_slice_36/stack_1.interactive_attention/strided_slice_36/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_37/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_37/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    	       
?
.interactive_attention/strided_slice_37/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_37StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_37/stack.interactive_attention/strided_slice_37/stack_1.interactive_attention/strided_slice_37/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_18Mul&interactive_attention/strided_slice_36&interactive_attention/strided_slice_37*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_38/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_38/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_38/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_38StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_38/stack.interactive_attention/strided_slice_38/stack_1.interactive_attention/strided_slice_38/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_39/stackConst*
_output_shapes
:*
dtype0*!
valueB"    	       
?
.interactive_attention/strided_slice_39/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    
       
?
.interactive_attention/strided_slice_39/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_39StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_39/stack.interactive_attention/strided_slice_39/stack_1.interactive_attention/strided_slice_39/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_19Mul&interactive_attention/strided_slice_38&interactive_attention/strided_slice_39*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_40/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_40/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_40/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_40StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_40/stack.interactive_attention/strided_slice_40/stack_1.interactive_attention/strided_slice_40/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_41/stackConst*
_output_shapes
:*
dtype0*!
valueB"    
       
?
.interactive_attention/strided_slice_41/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_41/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_41StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_41/stack.interactive_attention/strided_slice_41/stack_1.interactive_attention/strided_slice_41/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_20Mul&interactive_attention/strided_slice_40&interactive_attention/strided_slice_41*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_42/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_42/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_42/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_42StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_42/stack.interactive_attention/strided_slice_42/stack_1.interactive_attention/strided_slice_42/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_43/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_43/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_43/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_43StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_43/stack.interactive_attention/strided_slice_43/stack_1.interactive_attention/strided_slice_43/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_21Mul&interactive_attention/strided_slice_42&interactive_attention/strided_slice_43*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_44/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_44/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_44/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_44StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_44/stack.interactive_attention/strided_slice_44/stack_1.interactive_attention/strided_slice_44/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_45/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_45/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_45/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_45StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_45/stack.interactive_attention/strided_slice_45/stack_1.interactive_attention/strided_slice_45/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_22Mul&interactive_attention/strided_slice_44&interactive_attention/strided_slice_45*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_46/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_46/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_46/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_46StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_46/stack.interactive_attention/strided_slice_46/stack_1.interactive_attention/strided_slice_46/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_47/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_47/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_47/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_47StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_47/stack.interactive_attention/strided_slice_47/stack_1.interactive_attention/strided_slice_47/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_23Mul&interactive_attention/strided_slice_46&interactive_attention/strided_slice_47*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_48/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_48/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_48/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_48StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_48/stack.interactive_attention/strided_slice_48/stack_1.interactive_attention/strided_slice_48/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_49/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_49/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_49/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_49StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_49/stack.interactive_attention/strided_slice_49/stack_1.interactive_attention/strided_slice_49/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_24Mul&interactive_attention/strided_slice_48&interactive_attention/strided_slice_49*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_50/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_50/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_50/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_50StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_50/stack.interactive_attention/strided_slice_50/stack_1.interactive_attention/strided_slice_50/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_51/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_51/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_51/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_51StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_51/stack.interactive_attention/strided_slice_51/stack_1.interactive_attention/strided_slice_51/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_25Mul&interactive_attention/strided_slice_50&interactive_attention/strided_slice_51*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_52/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_52/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_52/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_52StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_52/stack.interactive_attention/strided_slice_52/stack_1.interactive_attention/strided_slice_52/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_53/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_53/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_53/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_53StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_53/stack.interactive_attention/strided_slice_53/stack_1.interactive_attention/strided_slice_53/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_26Mul&interactive_attention/strided_slice_52&interactive_attention/strided_slice_53*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_54/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_54/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_54/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_54StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_54/stack.interactive_attention/strided_slice_54/stack_1.interactive_attention/strided_slice_54/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_55/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_55/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_55/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_55StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_55/stack.interactive_attention/strided_slice_55/stack_1.interactive_attention/strided_slice_55/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_27Mul&interactive_attention/strided_slice_54&interactive_attention/strided_slice_55*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_56/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_56/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_56/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_56StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_56/stack.interactive_attention/strided_slice_56/stack_1.interactive_attention/strided_slice_56/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_57/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_57/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    	       
?
.interactive_attention/strided_slice_57/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_57StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_57/stack.interactive_attention/strided_slice_57/stack_1.interactive_attention/strided_slice_57/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_28Mul&interactive_attention/strided_slice_56&interactive_attention/strided_slice_57*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_58/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_58/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_58/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_58StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_58/stack.interactive_attention/strided_slice_58/stack_1.interactive_attention/strided_slice_58/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_59/stackConst*
_output_shapes
:*
dtype0*!
valueB"    	       
?
.interactive_attention/strided_slice_59/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    
       
?
.interactive_attention/strided_slice_59/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_59StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_59/stack.interactive_attention/strided_slice_59/stack_1.interactive_attention/strided_slice_59/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_29Mul&interactive_attention/strided_slice_58&interactive_attention/strided_slice_59*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_60/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_60/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_60/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_60StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_60/stack.interactive_attention/strided_slice_60/stack_1.interactive_attention/strided_slice_60/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_61/stackConst*
_output_shapes
:*
dtype0*!
valueB"    
       
?
.interactive_attention/strided_slice_61/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_61/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_61StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_61/stack.interactive_attention/strided_slice_61/stack_1.interactive_attention/strided_slice_61/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_30Mul&interactive_attention/strided_slice_60&interactive_attention/strided_slice_61*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_62/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_62/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_62/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_62StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_62/stack.interactive_attention/strided_slice_62/stack_1.interactive_attention/strided_slice_62/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_63/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_63/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_63/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_63StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_63/stack.interactive_attention/strided_slice_63/stack_1.interactive_attention/strided_slice_63/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_31Mul&interactive_attention/strided_slice_62&interactive_attention/strided_slice_63*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_64/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_64/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_64/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_64StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_64/stack.interactive_attention/strided_slice_64/stack_1.interactive_attention/strided_slice_64/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_65/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_65/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_65/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_65StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_65/stack.interactive_attention/strided_slice_65/stack_1.interactive_attention/strided_slice_65/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_32Mul&interactive_attention/strided_slice_64&interactive_attention/strided_slice_65*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_66/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_66/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_66/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_66StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_66/stack.interactive_attention/strided_slice_66/stack_1.interactive_attention/strided_slice_66/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_67/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_67/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_67/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_67StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_67/stack.interactive_attention/strided_slice_67/stack_1.interactive_attention/strided_slice_67/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_33Mul&interactive_attention/strided_slice_66&interactive_attention/strided_slice_67*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_68/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_68/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_68/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_68StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_68/stack.interactive_attention/strided_slice_68/stack_1.interactive_attention/strided_slice_68/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_69/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_69/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_69/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_69StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_69/stack.interactive_attention/strided_slice_69/stack_1.interactive_attention/strided_slice_69/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_34Mul&interactive_attention/strided_slice_68&interactive_attention/strided_slice_69*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_70/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_70/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_70/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_70StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_70/stack.interactive_attention/strided_slice_70/stack_1.interactive_attention/strided_slice_70/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_71/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_71/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_71/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_71StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_71/stack.interactive_attention/strided_slice_71/stack_1.interactive_attention/strided_slice_71/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_35Mul&interactive_attention/strided_slice_70&interactive_attention/strided_slice_71*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_72/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_72/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_72/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_72StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_72/stack.interactive_attention/strided_slice_72/stack_1.interactive_attention/strided_slice_72/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_73/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_73/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_73/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_73StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_73/stack.interactive_attention/strided_slice_73/stack_1.interactive_attention/strided_slice_73/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_36Mul&interactive_attention/strided_slice_72&interactive_attention/strided_slice_73*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_74/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_74/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_74/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_74StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_74/stack.interactive_attention/strided_slice_74/stack_1.interactive_attention/strided_slice_74/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_75/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_75/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    	       
?
.interactive_attention/strided_slice_75/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_75StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_75/stack.interactive_attention/strided_slice_75/stack_1.interactive_attention/strided_slice_75/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_37Mul&interactive_attention/strided_slice_74&interactive_attention/strided_slice_75*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_76/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_76/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_76/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_76StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_76/stack.interactive_attention/strided_slice_76/stack_1.interactive_attention/strided_slice_76/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_77/stackConst*
_output_shapes
:*
dtype0*!
valueB"    	       
?
.interactive_attention/strided_slice_77/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    
       
?
.interactive_attention/strided_slice_77/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_77StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_77/stack.interactive_attention/strided_slice_77/stack_1.interactive_attention/strided_slice_77/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_38Mul&interactive_attention/strided_slice_76&interactive_attention/strided_slice_77*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_78/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_78/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_78/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_78StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_78/stack.interactive_attention/strided_slice_78/stack_1.interactive_attention/strided_slice_78/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_79/stackConst*
_output_shapes
:*
dtype0*!
valueB"    
       
?
.interactive_attention/strided_slice_79/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_79/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_79StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_79/stack.interactive_attention/strided_slice_79/stack_1.interactive_attention/strided_slice_79/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_39Mul&interactive_attention/strided_slice_78&interactive_attention/strided_slice_79*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_80/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_80/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_80/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_80StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_80/stack.interactive_attention/strided_slice_80/stack_1.interactive_attention/strided_slice_80/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_81/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_81/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_81/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_81StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_81/stack.interactive_attention/strided_slice_81/stack_1.interactive_attention/strided_slice_81/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_40Mul&interactive_attention/strided_slice_80&interactive_attention/strided_slice_81*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_82/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_82/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_82/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_82StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_82/stack.interactive_attention/strided_slice_82/stack_1.interactive_attention/strided_slice_82/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_83/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_83/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_83/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_83StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_83/stack.interactive_attention/strided_slice_83/stack_1.interactive_attention/strided_slice_83/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_41Mul&interactive_attention/strided_slice_82&interactive_attention/strided_slice_83*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_84/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_84/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_84/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_84StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_84/stack.interactive_attention/strided_slice_84/stack_1.interactive_attention/strided_slice_84/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_85/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_85/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_85/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_85StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_85/stack.interactive_attention/strided_slice_85/stack_1.interactive_attention/strided_slice_85/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_42Mul&interactive_attention/strided_slice_84&interactive_attention/strided_slice_85*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_86/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_86/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_86/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_86StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_86/stack.interactive_attention/strided_slice_86/stack_1.interactive_attention/strided_slice_86/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_87/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_87/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_87/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_87StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_87/stack.interactive_attention/strided_slice_87/stack_1.interactive_attention/strided_slice_87/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_43Mul&interactive_attention/strided_slice_86&interactive_attention/strided_slice_87*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_88/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_88/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_88/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_88StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_88/stack.interactive_attention/strided_slice_88/stack_1.interactive_attention/strided_slice_88/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_89/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_89/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_89/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_89StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_89/stack.interactive_attention/strided_slice_89/stack_1.interactive_attention/strided_slice_89/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_44Mul&interactive_attention/strided_slice_88&interactive_attention/strided_slice_89*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_90/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_90/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_90/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_90StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_90/stack.interactive_attention/strided_slice_90/stack_1.interactive_attention/strided_slice_90/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_91/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_91/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    	       
?
.interactive_attention/strided_slice_91/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_91StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_91/stack.interactive_attention/strided_slice_91/stack_1.interactive_attention/strided_slice_91/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_45Mul&interactive_attention/strided_slice_90&interactive_attention/strided_slice_91*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_92/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_92/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_92/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_92StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_92/stack.interactive_attention/strided_slice_92/stack_1.interactive_attention/strided_slice_92/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_93/stackConst*
_output_shapes
:*
dtype0*!
valueB"    	       
?
.interactive_attention/strided_slice_93/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    
       
?
.interactive_attention/strided_slice_93/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_93StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_93/stack.interactive_attention/strided_slice_93/stack_1.interactive_attention/strided_slice_93/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_46Mul&interactive_attention/strided_slice_92&interactive_attention/strided_slice_93*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_94/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_94/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_94/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_94StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_94/stack.interactive_attention/strided_slice_94/stack_1.interactive_attention/strided_slice_94/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_95/stackConst*
_output_shapes
:*
dtype0*!
valueB"    
       
?
.interactive_attention/strided_slice_95/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_95/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_95StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_95/stack.interactive_attention/strided_slice_95/stack_1.interactive_attention/strided_slice_95/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_47Mul&interactive_attention/strided_slice_94&interactive_attention/strided_slice_95*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_96/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_96/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_96/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_96StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_96/stack.interactive_attention/strided_slice_96/stack_1.interactive_attention/strided_slice_96/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_97/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_97/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_97/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_97StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_97/stack.interactive_attention/strided_slice_97/stack_1.interactive_attention/strided_slice_97/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_48Mul&interactive_attention/strided_slice_96&interactive_attention/strided_slice_97*
T0*'
_output_shapes
:?????????

?
,interactive_attention/strided_slice_98/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_98/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_98/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_98StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_98/stack.interactive_attention/strided_slice_98/stack_1.interactive_attention/strided_slice_98/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
,interactive_attention/strided_slice_99/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_99/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
.interactive_attention/strided_slice_99/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
&interactive_attention/strided_slice_99StridedSliceembeddings/embeddings_out,interactive_attention/strided_slice_99/stack.interactive_attention/strided_slice_99/stack_1.interactive_attention/strided_slice_99/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_49Mul&interactive_attention/strided_slice_98&interactive_attention/strided_slice_99*
T0*'
_output_shapes
:?????????

?
-interactive_attention/strided_slice_100/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_100/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_100/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_100StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_100/stack/interactive_attention/strided_slice_100/stack_1/interactive_attention/strided_slice_100/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
-interactive_attention/strided_slice_101/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_101/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_101/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_101StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_101/stack/interactive_attention/strided_slice_101/stack_1/interactive_attention/strided_slice_101/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_50Mul'interactive_attention/strided_slice_100'interactive_attention/strided_slice_101*
T0*'
_output_shapes
:?????????

?
-interactive_attention/strided_slice_102/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_102/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_102/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_102StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_102/stack/interactive_attention/strided_slice_102/stack_1/interactive_attention/strided_slice_102/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
-interactive_attention/strided_slice_103/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_103/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_103/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_103StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_103/stack/interactive_attention/strided_slice_103/stack_1/interactive_attention/strided_slice_103/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_51Mul'interactive_attention/strided_slice_102'interactive_attention/strided_slice_103*
T0*'
_output_shapes
:?????????

?
-interactive_attention/strided_slice_104/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_104/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_104/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_104StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_104/stack/interactive_attention/strided_slice_104/stack_1/interactive_attention/strided_slice_104/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
-interactive_attention/strided_slice_105/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_105/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    	       
?
/interactive_attention/strided_slice_105/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_105StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_105/stack/interactive_attention/strided_slice_105/stack_1/interactive_attention/strided_slice_105/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_52Mul'interactive_attention/strided_slice_104'interactive_attention/strided_slice_105*
T0*'
_output_shapes
:?????????

?
-interactive_attention/strided_slice_106/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_106/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_106/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_106StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_106/stack/interactive_attention/strided_slice_106/stack_1/interactive_attention/strided_slice_106/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
-interactive_attention/strided_slice_107/stackConst*
_output_shapes
:*
dtype0*!
valueB"    	       
?
/interactive_attention/strided_slice_107/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    
       
?
/interactive_attention/strided_slice_107/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_107StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_107/stack/interactive_attention/strided_slice_107/stack_1/interactive_attention/strided_slice_107/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_53Mul'interactive_attention/strided_slice_106'interactive_attention/strided_slice_107*
T0*'
_output_shapes
:?????????

?
-interactive_attention/strided_slice_108/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_108/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_108/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_108StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_108/stack/interactive_attention/strided_slice_108/stack_1/interactive_attention/strided_slice_108/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
-interactive_attention/strided_slice_109/stackConst*
_output_shapes
:*
dtype0*!
valueB"    
       
?
/interactive_attention/strided_slice_109/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_109/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_109StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_109/stack/interactive_attention/strided_slice_109/stack_1/interactive_attention/strided_slice_109/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_54Mul'interactive_attention/strided_slice_108'interactive_attention/strided_slice_109*
T0*'
_output_shapes
:?????????

?
-interactive_attention/strided_slice_110/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_110/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_110/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_110StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_110/stack/interactive_attention/strided_slice_110/stack_1/interactive_attention/strided_slice_110/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
-interactive_attention/strided_slice_111/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_111/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_111/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_111StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_111/stack/interactive_attention/strided_slice_111/stack_1/interactive_attention/strided_slice_111/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_55Mul'interactive_attention/strided_slice_110'interactive_attention/strided_slice_111*
T0*'
_output_shapes
:?????????

?
-interactive_attention/strided_slice_112/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_112/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_112/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_112StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_112/stack/interactive_attention/strided_slice_112/stack_1/interactive_attention/strided_slice_112/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
-interactive_attention/strided_slice_113/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_113/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_113/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_113StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_113/stack/interactive_attention/strided_slice_113/stack_1/interactive_attention/strided_slice_113/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_56Mul'interactive_attention/strided_slice_112'interactive_attention/strided_slice_113*
T0*'
_output_shapes
:?????????

?
-interactive_attention/strided_slice_114/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_114/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_114/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_114StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_114/stack/interactive_attention/strided_slice_114/stack_1/interactive_attention/strided_slice_114/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
-interactive_attention/strided_slice_115/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_115/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_115/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_115StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_115/stack/interactive_attention/strided_slice_115/stack_1/interactive_attention/strided_slice_115/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_57Mul'interactive_attention/strided_slice_114'interactive_attention/strided_slice_115*
T0*'
_output_shapes
:?????????

?
-interactive_attention/strided_slice_116/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_116/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_116/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_116StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_116/stack/interactive_attention/strided_slice_116/stack_1/interactive_attention/strided_slice_116/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
-interactive_attention/strided_slice_117/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_117/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    	       
?
/interactive_attention/strided_slice_117/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_117StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_117/stack/interactive_attention/strided_slice_117/stack_1/interactive_attention/strided_slice_117/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_58Mul'interactive_attention/strided_slice_116'interactive_attention/strided_slice_117*
T0*'
_output_shapes
:?????????

?
-interactive_attention/strided_slice_118/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_118/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_118/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_118StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_118/stack/interactive_attention/strided_slice_118/stack_1/interactive_attention/strided_slice_118/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
-interactive_attention/strided_slice_119/stackConst*
_output_shapes
:*
dtype0*!
valueB"    	       
?
/interactive_attention/strided_slice_119/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    
       
?
/interactive_attention/strided_slice_119/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_119StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_119/stack/interactive_attention/strided_slice_119/stack_1/interactive_attention/strided_slice_119/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_59Mul'interactive_attention/strided_slice_118'interactive_attention/strided_slice_119*
T0*'
_output_shapes
:?????????

?
-interactive_attention/strided_slice_120/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_120/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_120/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_120StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_120/stack/interactive_attention/strided_slice_120/stack_1/interactive_attention/strided_slice_120/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
-interactive_attention/strided_slice_121/stackConst*
_output_shapes
:*
dtype0*!
valueB"    
       
?
/interactive_attention/strided_slice_121/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_121/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_121StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_121/stack/interactive_attention/strided_slice_121/stack_1/interactive_attention/strided_slice_121/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_60Mul'interactive_attention/strided_slice_120'interactive_attention/strided_slice_121*
T0*'
_output_shapes
:?????????

?
-interactive_attention/strided_slice_122/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_122/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_122/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_122StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_122/stack/interactive_attention/strided_slice_122/stack_1/interactive_attention/strided_slice_122/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
-interactive_attention/strided_slice_123/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_123/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_123/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_123StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_123/stack/interactive_attention/strided_slice_123/stack_1/interactive_attention/strided_slice_123/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_61Mul'interactive_attention/strided_slice_122'interactive_attention/strided_slice_123*
T0*'
_output_shapes
:?????????

?
-interactive_attention/strided_slice_124/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_124/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_124/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_124StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_124/stack/interactive_attention/strided_slice_124/stack_1/interactive_attention/strided_slice_124/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
-interactive_attention/strided_slice_125/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_125/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_125/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_125StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_125/stack/interactive_attention/strided_slice_125/stack_1/interactive_attention/strided_slice_125/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_62Mul'interactive_attention/strided_slice_124'interactive_attention/strided_slice_125*
T0*'
_output_shapes
:?????????

?
-interactive_attention/strided_slice_126/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_126/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_126/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_126StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_126/stack/interactive_attention/strided_slice_126/stack_1/interactive_attention/strided_slice_126/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
-interactive_attention/strided_slice_127/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_127/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    	       
?
/interactive_attention/strided_slice_127/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_127StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_127/stack/interactive_attention/strided_slice_127/stack_1/interactive_attention/strided_slice_127/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_63Mul'interactive_attention/strided_slice_126'interactive_attention/strided_slice_127*
T0*'
_output_shapes
:?????????

?
-interactive_attention/strided_slice_128/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_128/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_128/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_128StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_128/stack/interactive_attention/strided_slice_128/stack_1/interactive_attention/strided_slice_128/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
-interactive_attention/strided_slice_129/stackConst*
_output_shapes
:*
dtype0*!
valueB"    	       
?
/interactive_attention/strided_slice_129/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    
       
?
/interactive_attention/strided_slice_129/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_129StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_129/stack/interactive_attention/strided_slice_129/stack_1/interactive_attention/strided_slice_129/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_64Mul'interactive_attention/strided_slice_128'interactive_attention/strided_slice_129*
T0*'
_output_shapes
:?????????

?
-interactive_attention/strided_slice_130/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_130/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_130/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_130StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_130/stack/interactive_attention/strided_slice_130/stack_1/interactive_attention/strided_slice_130/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
-interactive_attention/strided_slice_131/stackConst*
_output_shapes
:*
dtype0*!
valueB"    
       
?
/interactive_attention/strided_slice_131/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_131/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_131StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_131/stack/interactive_attention/strided_slice_131/stack_1/interactive_attention/strided_slice_131/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_65Mul'interactive_attention/strided_slice_130'interactive_attention/strided_slice_131*
T0*'
_output_shapes
:?????????

?
-interactive_attention/strided_slice_132/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_132/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_132/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_132StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_132/stack/interactive_attention/strided_slice_132/stack_1/interactive_attention/strided_slice_132/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
-interactive_attention/strided_slice_133/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_133/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_133/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_133StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_133/stack/interactive_attention/strided_slice_133/stack_1/interactive_attention/strided_slice_133/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_66Mul'interactive_attention/strided_slice_132'interactive_attention/strided_slice_133*
T0*'
_output_shapes
:?????????

?
-interactive_attention/strided_slice_134/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_134/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_134/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_134StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_134/stack/interactive_attention/strided_slice_134/stack_1/interactive_attention/strided_slice_134/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
-interactive_attention/strided_slice_135/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_135/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_135/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_135StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_135/stack/interactive_attention/strided_slice_135/stack_1/interactive_attention/strided_slice_135/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_67Mul'interactive_attention/strided_slice_134'interactive_attention/strided_slice_135*
T0*'
_output_shapes
:?????????

?
-interactive_attention/strided_slice_136/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_136/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    	       
?
/interactive_attention/strided_slice_136/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_136StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_136/stack/interactive_attention/strided_slice_136/stack_1/interactive_attention/strided_slice_136/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
-interactive_attention/strided_slice_137/stackConst*
_output_shapes
:*
dtype0*!
valueB"    	       
?
/interactive_attention/strided_slice_137/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    
       
?
/interactive_attention/strided_slice_137/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_137StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_137/stack/interactive_attention/strided_slice_137/stack_1/interactive_attention/strided_slice_137/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_68Mul'interactive_attention/strided_slice_136'interactive_attention/strided_slice_137*
T0*'
_output_shapes
:?????????

?
-interactive_attention/strided_slice_138/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_138/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    	       
?
/interactive_attention/strided_slice_138/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_138StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_138/stack/interactive_attention/strided_slice_138/stack_1/interactive_attention/strided_slice_138/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
-interactive_attention/strided_slice_139/stackConst*
_output_shapes
:*
dtype0*!
valueB"    
       
?
/interactive_attention/strided_slice_139/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_139/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_139StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_139/stack/interactive_attention/strided_slice_139/stack_1/interactive_attention/strided_slice_139/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_69Mul'interactive_attention/strided_slice_138'interactive_attention/strided_slice_139*
T0*'
_output_shapes
:?????????

?
-interactive_attention/strided_slice_140/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_140/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    	       
?
/interactive_attention/strided_slice_140/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_140StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_140/stack/interactive_attention/strided_slice_140/stack_1/interactive_attention/strided_slice_140/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
-interactive_attention/strided_slice_141/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_141/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_141/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_141StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_141/stack/interactive_attention/strided_slice_141/stack_1/interactive_attention/strided_slice_141/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_70Mul'interactive_attention/strided_slice_140'interactive_attention/strided_slice_141*
T0*'
_output_shapes
:?????????

?
-interactive_attention/strided_slice_142/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_142/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    	       
?
/interactive_attention/strided_slice_142/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_142StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_142/stack/interactive_attention/strided_slice_142/stack_1/interactive_attention/strided_slice_142/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
-interactive_attention/strided_slice_143/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_143/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_143/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_143StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_143/stack/interactive_attention/strided_slice_143/stack_1/interactive_attention/strided_slice_143/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_71Mul'interactive_attention/strided_slice_142'interactive_attention/strided_slice_143*
T0*'
_output_shapes
:?????????

?
-interactive_attention/strided_slice_144/stackConst*
_output_shapes
:*
dtype0*!
valueB"    	       
?
/interactive_attention/strided_slice_144/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    
       
?
/interactive_attention/strided_slice_144/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_144StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_144/stack/interactive_attention/strided_slice_144/stack_1/interactive_attention/strided_slice_144/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
-interactive_attention/strided_slice_145/stackConst*
_output_shapes
:*
dtype0*!
valueB"    
       
?
/interactive_attention/strided_slice_145/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_145/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_145StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_145/stack/interactive_attention/strided_slice_145/stack_1/interactive_attention/strided_slice_145/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_72Mul'interactive_attention/strided_slice_144'interactive_attention/strided_slice_145*
T0*'
_output_shapes
:?????????

?
-interactive_attention/strided_slice_146/stackConst*
_output_shapes
:*
dtype0*!
valueB"    	       
?
/interactive_attention/strided_slice_146/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    
       
?
/interactive_attention/strided_slice_146/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_146StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_146/stack/interactive_attention/strided_slice_146/stack_1/interactive_attention/strided_slice_146/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
-interactive_attention/strided_slice_147/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_147/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_147/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_147StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_147/stack/interactive_attention/strided_slice_147/stack_1/interactive_attention/strided_slice_147/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_73Mul'interactive_attention/strided_slice_146'interactive_attention/strided_slice_147*
T0*'
_output_shapes
:?????????

?
-interactive_attention/strided_slice_148/stackConst*
_output_shapes
:*
dtype0*!
valueB"    	       
?
/interactive_attention/strided_slice_148/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    
       
?
/interactive_attention/strided_slice_148/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_148StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_148/stack/interactive_attention/strided_slice_148/stack_1/interactive_attention/strided_slice_148/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
-interactive_attention/strided_slice_149/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_149/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_149/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_149StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_149/stack/interactive_attention/strided_slice_149/stack_1/interactive_attention/strided_slice_149/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_74Mul'interactive_attention/strided_slice_148'interactive_attention/strided_slice_149*
T0*'
_output_shapes
:?????????

?
-interactive_attention/strided_slice_150/stackConst*
_output_shapes
:*
dtype0*!
valueB"    
       
?
/interactive_attention/strided_slice_150/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_150/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_150StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_150/stack/interactive_attention/strided_slice_150/stack_1/interactive_attention/strided_slice_150/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
-interactive_attention/strided_slice_151/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_151/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_151/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_151StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_151/stack/interactive_attention/strided_slice_151/stack_1/interactive_attention/strided_slice_151/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_75Mul'interactive_attention/strided_slice_150'interactive_attention/strided_slice_151*
T0*'
_output_shapes
:?????????

?
-interactive_attention/strided_slice_152/stackConst*
_output_shapes
:*
dtype0*!
valueB"    
       
?
/interactive_attention/strided_slice_152/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_152/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_152StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_152/stack/interactive_attention/strided_slice_152/stack_1/interactive_attention/strided_slice_152/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
-interactive_attention/strided_slice_153/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_153/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_153/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_153StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_153/stack/interactive_attention/strided_slice_153/stack_1/interactive_attention/strided_slice_153/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_76Mul'interactive_attention/strided_slice_152'interactive_attention/strided_slice_153*
T0*'
_output_shapes
:?????????

?
-interactive_attention/strided_slice_154/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_154/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_154/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_154StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_154/stack/interactive_attention/strided_slice_154/stack_1/interactive_attention/strided_slice_154/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
-interactive_attention/strided_slice_155/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_155/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?
/interactive_attention/strided_slice_155/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
?
'interactive_attention/strided_slice_155StridedSliceembeddings/embeddings_out-interactive_attention/strided_slice_155/stack/interactive_attention/strided_slice_155/stack_1/interactive_attention/strided_slice_155/stack_2*
Index0*
T0*'
_output_shapes
:?????????
*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask
?
interactive_attention/Mul_77Mul'interactive_attention/strided_slice_154'interactive_attention/strided_slice_155*
T0*'
_output_shapes
:?????????

?
interactive_attention/stackPackinteractive_attention/Mulinteractive_attention/Mul_1interactive_attention/Mul_2interactive_attention/Mul_3interactive_attention/Mul_4interactive_attention/Mul_5interactive_attention/Mul_6interactive_attention/Mul_7interactive_attention/Mul_8interactive_attention/Mul_9interactive_attention/Mul_10interactive_attention/Mul_11interactive_attention/Mul_12interactive_attention/Mul_13interactive_attention/Mul_14interactive_attention/Mul_15interactive_attention/Mul_16interactive_attention/Mul_17interactive_attention/Mul_18interactive_attention/Mul_19interactive_attention/Mul_20interactive_attention/Mul_21interactive_attention/Mul_22interactive_attention/Mul_23interactive_attention/Mul_24interactive_attention/Mul_25interactive_attention/Mul_26interactive_attention/Mul_27interactive_attention/Mul_28interactive_attention/Mul_29interactive_attention/Mul_30interactive_attention/Mul_31interactive_attention/Mul_32interactive_attention/Mul_33interactive_attention/Mul_34interactive_attention/Mul_35interactive_attention/Mul_36interactive_attention/Mul_37interactive_attention/Mul_38interactive_attention/Mul_39interactive_attention/Mul_40interactive_attention/Mul_41interactive_attention/Mul_42interactive_attention/Mul_43interactive_attention/Mul_44interactive_attention/Mul_45interactive_attention/Mul_46interactive_attention/Mul_47interactive_attention/Mul_48interactive_attention/Mul_49interactive_attention/Mul_50interactive_attention/Mul_51interactive_attention/Mul_52interactive_attention/Mul_53interactive_attention/Mul_54interactive_attention/Mul_55interactive_attention/Mul_56interactive_attention/Mul_57interactive_attention/Mul_58interactive_attention/Mul_59interactive_attention/Mul_60interactive_attention/Mul_61interactive_attention/Mul_62interactive_attention/Mul_63interactive_attention/Mul_64interactive_attention/Mul_65interactive_attention/Mul_66interactive_attention/Mul_67interactive_attention/Mul_68interactive_attention/Mul_69interactive_attention/Mul_70interactive_attention/Mul_71interactive_attention/Mul_72interactive_attention/Mul_73interactive_attention/Mul_74interactive_attention/Mul_75interactive_attention/Mul_76interactive_attention/Mul_77*
NN*
T0*+
_output_shapes
:N?????????
*

axis 
y
$interactive_attention/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
?
interactive_attention/transpose	Transposeinteractive_attention/stack$interactive_attention/transpose/perm*
T0*
Tperm0*+
_output_shapes
:?????????N

m
+interactive_attention/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
?
interactive_attention/SumSuminteractive_attention/transpose+interactive_attention/Sum/reduction_indices*
T0*

Tidx0*'
_output_shapes
:?????????N*
	keep_dims( 
?
.attention_w/Initializer/truncated_normal/shapeConst*
_class
loc:@attention_w*
_output_shapes
:*
dtype0*
valueB"
      
?
-attention_w/Initializer/truncated_normal/meanConst*
_class
loc:@attention_w*
_output_shapes
: *
dtype0*
valueB
 *    
?
/attention_w/Initializer/truncated_normal/stddevConst*
_class
loc:@attention_w*
_output_shapes
: *
dtype0*
valueB
 *??>
?
8attention_w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal.attention_w/Initializer/truncated_normal/shape*
T0*
_class
loc:@attention_w*
_output_shapes

:
*
dtype0*
seed?*
seed2
?
,attention_w/Initializer/truncated_normal/mulMul8attention_w/Initializer/truncated_normal/TruncatedNormal/attention_w/Initializer/truncated_normal/stddev*
T0*
_class
loc:@attention_w*
_output_shapes

:

?
(attention_w/Initializer/truncated_normalAdd,attention_w/Initializer/truncated_normal/mul-attention_w/Initializer/truncated_normal/mean*
T0*
_class
loc:@attention_w*
_output_shapes

:

?
attention_w
VariableV2*
_class
loc:@attention_w*
_output_shapes

:
*
	container *
dtype0*
shape
:
*
shared_name 
?
attention_w/AssignAssignattention_w(attention_w/Initializer/truncated_normal*
T0*
_class
loc:@attention_w*
_output_shapes

:
*
use_locking(*
validate_shape(
r
attention_w/readIdentityattention_w*
T0*
_class
loc:@attention_w*
_output_shapes

:

t
#interactive_attention/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   
?
interactive_attention/ReshapeReshapeinteractive_attention/transpose#interactive_attention/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:?????????

?
interactive_attention/MatMulMatMulinteractive_attention/Reshapeattention_w/read*
T0*'
_output_shapes
:?????????*
transpose_a( *
transpose_b( 
z
%interactive_attention/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????N      
?
interactive_attention/Reshape_1Reshapeinteractive_attention/MatMul%interactive_attention/Reshape_1/shape*
T0*
Tshape0*+
_output_shapes
:?????????N
y
interactive_attention/TanhTanhinteractive_attention/Reshape_1*
T0*+
_output_shapes
:?????????N
?
-dense/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@dense/kernel*
_output_shapes
:*
dtype0*
valueB"      
?
+dense/kernel/Initializer/random_uniform/minConst*
_class
loc:@dense/kernel*
_output_shapes
: *
dtype0*
valueB
 *?Q?
?
+dense/kernel/Initializer/random_uniform/maxConst*
_class
loc:@dense/kernel*
_output_shapes
: *
dtype0*
valueB
 *?Q?
?
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
T0*
_class
loc:@dense/kernel*
_output_shapes

:*
dtype0*
seed?*
seed2
?
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 
?
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@dense/kernel*
_output_shapes

:
?
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes

:
?
dense/kernel
VariableV2*
_class
loc:@dense/kernel*
_output_shapes

:*
	container *
dtype0*
shape
:*
shared_name 
?
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
T0*
_class
loc:@dense/kernel*
_output_shapes

:*
use_locking(*
validate_shape(
u
dense/kernel/readIdentitydense/kernel*
T0*
_class
loc:@dense/kernel*
_output_shapes

:
?
dense/bias/Initializer/zerosConst*
_class
loc:@dense/bias*
_output_shapes
:*
dtype0*
valueB*    
?

dense/bias
VariableV2*
_class
loc:@dense/bias*
_output_shapes
:*
	container *
dtype0*
shape:*
shared_name 
?
dense/bias/AssignAssign
dense/biasdense/bias/Initializer/zeros*
T0*
_class
loc:@dense/bias*
_output_shapes
:*
use_locking(*
validate_shape(
k
dense/bias/readIdentity
dense/bias*
T0*
_class
loc:@dense/bias*
_output_shapes
:
t
*interactive_attention/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
{
*interactive_attention/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
?
+interactive_attention/dense/Tensordot/ShapeShapeinteractive_attention/Tanh*
T0*
_output_shapes
:*
out_type0
u
3interactive_attention/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
.interactive_attention/dense/Tensordot/GatherV2GatherV2+interactive_attention/dense/Tensordot/Shape*interactive_attention/dense/Tensordot/free3interactive_attention/dense/Tensordot/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:*

batch_dims 
w
5interactive_attention/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
0interactive_attention/dense/Tensordot/GatherV2_1GatherV2+interactive_attention/dense/Tensordot/Shape*interactive_attention/dense/Tensordot/axes5interactive_attention/dense/Tensordot/GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:*

batch_dims 
u
+interactive_attention/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
?
*interactive_attention/dense/Tensordot/ProdProd.interactive_attention/dense/Tensordot/GatherV2+interactive_attention/dense/Tensordot/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
w
-interactive_attention/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
?
,interactive_attention/dense/Tensordot/Prod_1Prod0interactive_attention/dense/Tensordot/GatherV2_1-interactive_attention/dense/Tensordot/Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
s
1interactive_attention/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
,interactive_attention/dense/Tensordot/concatConcatV2*interactive_attention/dense/Tensordot/free*interactive_attention/dense/Tensordot/axes1interactive_attention/dense/Tensordot/concat/axis*
N*
T0*

Tidx0*
_output_shapes
:
?
+interactive_attention/dense/Tensordot/stackPack*interactive_attention/dense/Tensordot/Prod,interactive_attention/dense/Tensordot/Prod_1*
N*
T0*
_output_shapes
:*

axis 
?
/interactive_attention/dense/Tensordot/transpose	Transposeinteractive_attention/Tanh,interactive_attention/dense/Tensordot/concat*
T0*
Tperm0*+
_output_shapes
:?????????N
?
-interactive_attention/dense/Tensordot/ReshapeReshape/interactive_attention/dense/Tensordot/transpose+interactive_attention/dense/Tensordot/stack*
T0*
Tshape0*0
_output_shapes
:??????????????????
?
,interactive_attention/dense/Tensordot/MatMulMatMul-interactive_attention/dense/Tensordot/Reshapedense/kernel/read*
T0*'
_output_shapes
:?????????*
transpose_a( *
transpose_b( 
w
-interactive_attention/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:
u
3interactive_attention/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
.interactive_attention/dense/Tensordot/concat_1ConcatV2.interactive_attention/dense/Tensordot/GatherV2-interactive_attention/dense/Tensordot/Const_23interactive_attention/dense/Tensordot/concat_1/axis*
N*
T0*

Tidx0*
_output_shapes
:
?
%interactive_attention/dense/TensordotReshape,interactive_attention/dense/Tensordot/MatMul.interactive_attention/dense/Tensordot/concat_1*
T0*
Tshape0*+
_output_shapes
:?????????N
?
#interactive_attention/dense/BiasAddBiasAdd%interactive_attention/dense/Tensordotdense/bias/read*
T0*+
_output_shapes
:?????????N*
data_formatNHWC
\
interactive_attention/RankConst*
_output_shapes
: *
dtype0*
value	B :
^
interactive_attention/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :
]
interactive_attention/Sub/yConst*
_output_shapes
: *
dtype0*
value	B :
|
interactive_attention/SubSubinteractive_attention/Rank_1interactive_attention/Sub/y*
T0*
_output_shapes
: 
c
!interactive_attention/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
c
!interactive_attention/range/limitConst*
_output_shapes
: *
dtype0*
value	B :
c
!interactive_attention/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
?
interactive_attention/rangeRange!interactive_attention/range/start!interactive_attention/range/limit!interactive_attention/range/delta*

Tidx0*
_output_shapes
:
e
#interactive_attention/range_1/startConst*
_output_shapes
: *
dtype0*
value	B :
e
#interactive_attention/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :
?
interactive_attention/range_1Range#interactive_attention/range_1/startinteractive_attention/Sub#interactive_attention/range_1/delta*

Tidx0*
_output_shapes
: 
?
%interactive_attention/concat/values_1Packinteractive_attention/Sub*
N*
T0*
_output_shapes
:*

axis 
o
%interactive_attention/concat/values_3Const*
_output_shapes
:*
dtype0*
valueB:
c
!interactive_attention/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
interactive_attention/concatConcatV2interactive_attention/range%interactive_attention/concat/values_1interactive_attention/range_1%interactive_attention/concat/values_3!interactive_attention/concat/axis*
N*
T0*

Tidx0*
_output_shapes
:
?
!interactive_attention/transpose_1	Transpose#interactive_attention/dense/BiasAddinteractive_attention/concat*
T0*
Tperm0*+
_output_shapes
:?????????N
?
interactive_attention/SoftmaxSoftmax!interactive_attention/transpose_1*
T0*+
_output_shapes
:?????????N
_
interactive_attention/Sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :
?
interactive_attention/Sub_1Subinteractive_attention/Rank_1interactive_attention/Sub_1/y*
T0*
_output_shapes
: 
e
#interactive_attention/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 
e
#interactive_attention/range_2/limitConst*
_output_shapes
: *
dtype0*
value	B :
e
#interactive_attention/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :
?
interactive_attention/range_2Range#interactive_attention/range_2/start#interactive_attention/range_2/limit#interactive_attention/range_2/delta*

Tidx0*
_output_shapes
:
e
#interactive_attention/range_3/startConst*
_output_shapes
: *
dtype0*
value	B :
e
#interactive_attention/range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :
?
interactive_attention/range_3Range#interactive_attention/range_3/startinteractive_attention/Sub_1#interactive_attention/range_3/delta*

Tidx0*
_output_shapes
: 
?
'interactive_attention/concat_1/values_1Packinteractive_attention/Sub_1*
N*
T0*
_output_shapes
:*

axis 
q
'interactive_attention/concat_1/values_3Const*
_output_shapes
:*
dtype0*
valueB:
e
#interactive_attention/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
interactive_attention/concat_1ConcatV2interactive_attention/range_2'interactive_attention/concat_1/values_1interactive_attention/range_3'interactive_attention/concat_1/values_3#interactive_attention/concat_1/axis*
N*
T0*

Tidx0*
_output_shapes
:
?
&interactive_attention/attention_matrix	Transposeinteractive_attention/Softmaxinteractive_attention/concat_1*
T0*
Tperm0*+
_output_shapes
:?????????N
h
#interactive_attention/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
?
!interactive_attention/dropout/MulMul&interactive_attention/attention_matrix#interactive_attention/dropout/Const*
T0*+
_output_shapes
:?????????N
?
#interactive_attention/dropout/ShapeShape&interactive_attention/attention_matrix*
T0*
_output_shapes
:*
out_type0
?
:interactive_attention/dropout/random_uniform/RandomUniformRandomUniform#interactive_attention/dropout/Shape*
T0*+
_output_shapes
:?????????N*
dtype0*
seed?*
seed2
q
,interactive_attention/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
?
*interactive_attention/dropout/GreaterEqualGreaterEqual:interactive_attention/dropout/random_uniform/RandomUniform,interactive_attention/dropout/GreaterEqual/y*
T0*+
_output_shapes
:?????????N
?
"interactive_attention/dropout/CastCast*interactive_attention/dropout/GreaterEqual*

DstT0*

SrcT0
*
Truncate( *+
_output_shapes
:?????????N
?
#interactive_attention/dropout/Mul_1Mul!interactive_attention/dropout/Mul"interactive_attention/dropout/Cast*
T0*+
_output_shapes
:?????????N
?
interactive_attention/Mul_78Mul#interactive_attention/dropout/Mul_1interactive_attention/transpose*
T0*+
_output_shapes
:?????????N

o
-interactive_attention/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
?
interactive_attention/Sum_1Suminteractive_attention/Mul_78-interactive_attention/Sum_1/reduction_indices*
T0*

Tidx0*'
_output_shapes
:?????????
*
	keep_dims( 
?
/dense_1/kernel/Initializer/random_uniform/shapeConst*!
_class
loc:@dense_1/kernel*
_output_shapes
:*
dtype0*
valueB"
      
?
-dense_1/kernel/Initializer/random_uniform/minConst*!
_class
loc:@dense_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *?=?
?
-dense_1/kernel/Initializer/random_uniform/maxConst*!
_class
loc:@dense_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *?=?
?
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:
*
dtype0*
seed?*
seed2
?
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
?
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:

?
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:

?
dense_1/kernel
VariableV2*!
_class
loc:@dense_1/kernel*
_output_shapes

:
*
	container *
dtype0*
shape
:
*
shared_name 
?
dense_1/kernel/AssignAssigndense_1/kernel)dense_1/kernel/Initializer/random_uniform*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:
*
use_locking(*
validate_shape(
{
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:

?
dense_1/bias/Initializer/zerosConst*
_class
loc:@dense_1/bias*
_output_shapes
:*
dtype0*
valueB*    
?
dense_1/bias
VariableV2*
_class
loc:@dense_1/bias*
_output_shapes
:*
	container *
dtype0*
shape:*
shared_name 
?
dense_1/bias/AssignAssigndense_1/biasdense_1/bias/Initializer/zeros*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:*
use_locking(*
validate_shape(
q
dense_1/bias/readIdentitydense_1/bias*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:
?
$interactive_attention/dense_1/MatMulMatMulinteractive_attention/Sum_1dense_1/kernel/read*
T0*'
_output_shapes
:?????????*
transpose_a( *
transpose_b( 
?
%interactive_attention/dense_1/BiasAddBiasAdd$interactive_attention/dense_1/MatMuldense_1/bias/read*
T0*'
_output_shapes
:?????????*
data_formatNHWC
m
IdentityIdentity%interactive_attention/dense_1/BiasAdd*
T0*'
_output_shapes
:?????????
L
scoreSigmoidIdentity*
T0*'
_output_shapes
:?????????

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
Y
save/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
dtype0*
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
dtype0*
shape: 
{
save/StaticRegexFullMatchStaticRegexFullMatch
save/Const"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*
a
save/Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part
f
save/Const_2Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
|
save/SelectSelectsave/StaticRegexFullMatchsave/Const_1save/Const_2"/device:CPU:**
T0*
_output_shapes
: 
w
save/StringJoin
StringJoin
save/Constsave/Select"/device:CPU:**
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
k
save/ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
?
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
?
save/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*z
valueqBoBattention_wB
dense/biasBdense/kernelBdense_1/biasBdense_1/kernelBembeddings/VariableBglobal_step
?
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 
?
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesattention_w
dense/biasdense/kerneldense_1/biasdense_1/kernelembeddings/Variableglobal_step"/device:CPU:0*
dtypes
	2	
?
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
?
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
N*
T0*
_output_shapes
:*

axis 
?
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(
?
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
?
save/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*z
valueqBoBattention_wB
dense/biasBdense/kernelBdense_1/biasBdense_1/kernelBembeddings/VariableBglobal_step
?
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	2	
?
save/AssignAssignattention_wsave/RestoreV2*
T0*
_class
loc:@attention_w*
_output_shapes

:
*
use_locking(*
validate_shape(
?
save/Assign_1Assign
dense/biassave/RestoreV2:1*
T0*
_class
loc:@dense/bias*
_output_shapes
:*
use_locking(*
validate_shape(
?
save/Assign_2Assigndense/kernelsave/RestoreV2:2*
T0*
_class
loc:@dense/kernel*
_output_shapes

:*
use_locking(*
validate_shape(
?
save/Assign_3Assigndense_1/biassave/RestoreV2:3*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:*
use_locking(*
validate_shape(
?
save/Assign_4Assigndense_1/kernelsave/RestoreV2:4*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:
*
use_locking(*
validate_shape(
?
save/Assign_5Assignembeddings/Variablesave/RestoreV2:5*
T0*&
_class
loc:@embeddings/Variable*
_output_shapes
:	?
*
use_locking(*
validate_shape(
?
save/Assign_6Assignglobal_stepsave/RestoreV2:6*
T0	*
_class
loc:@global_step*
_output_shapes
: *
use_locking(*
validate_shape(
?
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6
-
save/restore_allNoOp^save/restore_shard"?<
save/Const:0save/Identity:0save/restore_all (5 @F8"m
global_step^\
Z
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0H"%
saved_model_main_op


group_deps"?
trainable_variables??
m
embeddings/Variable:0embeddings/Variable/Assignembeddings/Variable/read:02embeddings/random_normal:08
e
attention_w:0attention_w/Assignattention_w/read:02*attention_w/Initializer/truncated_normal:08
g
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:08
V
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:08
o
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02+dense_1/kernel/Initializer/random_uniform:08
^
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:08"?
	variables??
Z
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0H
m
embeddings/Variable:0embeddings/Variable/Assignembeddings/Variable/read:02embeddings/random_normal:08
e
attention_w:0attention_w/Assignattention_w/read:02*attention_w/Initializer/truncated_normal:08
g
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:08
V
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:08
o
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02+dense_1/kernel/Initializer/random_uniform:08
^
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:08*?
serving_default?
!
Xi
Xi:0?????????
!
Xv
Xv:0?????????(
output
score:0?????????tensorflow/serving/predict