package gpu

import (
	"fmt"
	"math"
	"strings"

	"github.com/jmwri/neatgo/v2/network"
)

// The kernel is PTX, NVIDIA's portable assembly, that the driver compiles for
// whatever GPU it finds. Writing it directly, rather than CUDA C, is what lets
// this package work with nothing but the driver installed: no CUDA toolkit, no
// nvcc, no C compiler.
//
// One thread computes one network on one input sample. Threads are numbered so
// that neighbours share a network, which keeps a warp on the same code path: it
// reads the same node table and takes the same branch for each node's
// activation, and only the data differs.
//
// A network's node values live in a scratch buffer in global memory, laid out
// node-major so that the threads of a warp read adjacent words.

// Opcodes the kernel dispatches on. Anything below opBias is an activation
// function; opBias and opInput are the two node kinds that have none.
const (
	opBias  = 64
	opInput = 128 // plus the input position
)

// activationOps numbers the built-in activation functions. The kernel switches
// on these, so they are part of the encoding, not merely a lookup.
var activationOps = map[network.ActivationFunctionName]int32{
	network.NoActivation: 0,
	network.Identity:     1,
	network.Sigmoid:      2,
	network.Tanh:         3,
	network.Sin:          4,
	network.Gauss:        5,
	network.Relu:         6,
	network.Elu:          7,
	network.Lelu:         8,
	network.Selu:         9,
	network.SoftPlus:     10,
	network.Clamped:      11,
	network.Inv:          12,
	network.Log:          13,
	network.Exp:          14,
	network.Abs:          15,
	network.Hat:          16,
	network.Square:       17,
	network.Cube:         18,
}

// f32 renders x as a PTX single-precision literal.
func f32(x float64) string {
	return fmt.Sprintf("0f%08X", math.Float32bits(float32(x)))
}

// activationBodies holds the PTX for each activation, in the order of
// activationOps. Each reads the pre-activation sum from %f2 and leaves the
// result in %f5, with %f6-%f9 free as scratch. They mirror the Go functions in
// package network, using the GPU's approximate exp, log, sin and reciprocal
// instructions in place of the exact ones.
//
// ex2 computes 2^x, so e^x is ex2(x*log2(e)); the same identity is folded into
// the constants where a function scales its argument first.
func activationBodies() [19]string {
	log2e := math.Log2E
	ln2 := math.Ln2

	// clampF applies min/max against literals: dst = clamp(src, lo, hi).
	clampF := func(dst, src string, lo, hi float64) string {
		return fmt.Sprintf("\tmax.f32 %s, %s, %s;\n\tmin.f32 %s, %s, %s;\n", dst, src, f32(lo), dst, dst, f32(hi))
	}
	// exp puts e^src in dst.
	exp := func(dst, src string) string {
		return fmt.Sprintf("\tmul.f32 %s, %s, %s;\n\tex2.approx.ftz.f32 %s, %s;\n", dst, src, f32(log2e), dst, dst)
	}

	var b [19]string
	b[0] = "\tmov.f32 %f5, %f2;\n"
	b[1] = b[0]

	// sigmoid: 1 / (1 + e^-clamp(5x))
	b[2] = fmt.Sprintf("\tmul.f32 %%f6, %%f2, %s;\n", f32(5)) +
		clampF("%f6", "%f6", -60, 60) +
		"\tneg.f32 %f6, %f6;\n" +
		exp("%f7", "%f6") +
		fmt.Sprintf("\tadd.f32 %%f7, %%f7, %s;\n", f32(1)) +
		"\trcp.rn.f32 %f5, %f7;\n"

	// tanh: 1 - 2 / (e^(2y) + 1), y = clamp(2.5x). Written through exp because
	// that works on every GPU generation, unlike the tanh instruction. e^(2y)
	// overflows to +Inf at the clamp, and 2/Inf is 0, so the limits come out
	// exactly right.
	b[3] = fmt.Sprintf("\tmul.f32 %%f6, %%f2, %s;\n", f32(2.5)) +
		clampF("%f6", "%f6", -60, 60) +
		fmt.Sprintf("\tmul.f32 %%f6, %%f6, %s;\n\tex2.approx.ftz.f32 %%f7, %%f6;\n", f32(2*log2e)) +
		fmt.Sprintf("\tadd.f32 %%f7, %%f7, %s;\n", f32(1)) +
		"\trcp.rn.f32 %f8, %f7;\n" +
		fmt.Sprintf("\tfma.rn.f32 %%f5, %%f8, %s, %s;\n", f32(-2), f32(1))

	// sin(clamp(5x))
	b[4] = fmt.Sprintf("\tmul.f32 %%f6, %%f2, %s;\n", f32(5)) +
		clampF("%f6", "%f6", -60, 60) +
		"\tsin.approx.ftz.f32 %f5, %f6;\n"

	// gauss: e^(-5 clamp(x, ±3.4)^2)
	b[5] = clampF("%f6", "%f2", -3.4, 3.4) +
		"\tmul.f32 %f6, %f6, %f6;\n" +
		fmt.Sprintf("\tmul.f32 %%f6, %%f6, %s;\n\tex2.approx.ftz.f32 %%f5, %%f6;\n", f32(-5*log2e))

	// relu
	b[6] = fmt.Sprintf("\tmax.f32 %%f5, %%f2, %s;\n", f32(0))

	// elu: x if x > 0, else e^x - 1
	b[7] = fmt.Sprintf("\tsetp.gt.f32 %%p6, %%f2, %s;\n", f32(0)) +
		exp("%f7", "%f2") +
		fmt.Sprintf("\tsub.f32 %%f7, %%f7, %s;\n", f32(1)) +
		"\tselp.f32 %f5, %f2, %f7, %p6;\n"

	// lelu: x if x > 0, else 0.005x
	b[8] = fmt.Sprintf("\tsetp.gt.f32 %%p6, %%f2, %s;\n", f32(0)) +
		fmt.Sprintf("\tmul.f32 %%f7, %%f2, %s;\n", f32(.005)) +
		"\tselp.f32 %f5, %f2, %f7, %p6;\n"

	// selu
	const lambda, alpha = 1.0507009873554804934193349852946, 1.6732632423543772848170429916717
	b[9] = fmt.Sprintf("\tsetp.gt.f32 %%p6, %%f2, %s;\n", f32(0)) +
		exp("%f7", "%f2") +
		fmt.Sprintf("\tsub.f32 %%f7, %%f7, %s;\n", f32(1)) +
		fmt.Sprintf("\tmul.f32 %%f7, %%f7, %s;\n", f32(lambda*alpha)) +
		fmt.Sprintf("\tmul.f32 %%f8, %%f2, %s;\n", f32(lambda)) +
		"\tselp.f32 %f5, %f8, %f7, %p6;\n"

	// softplus: 0.2 ln(1 + e^clamp(5x))
	b[10] = fmt.Sprintf("\tmul.f32 %%f6, %%f2, %s;\n", f32(5)) +
		clampF("%f6", "%f6", -60, 60) +
		exp("%f7", "%f6") +
		fmt.Sprintf("\tadd.f32 %%f7, %%f7, %s;\n", f32(1)) +
		"\tlg2.approx.ftz.f32 %f8, %f7;\n" +
		fmt.Sprintf("\tmul.f32 %%f5, %%f8, %s;\n", f32(.2*ln2))

	// clamped
	b[11] = clampF("%f5", "%f2", -1, 1)

	// inv: 1/x, and 0 at x == 0
	b[12] = fmt.Sprintf("\tsetp.eq.f32 %%p6, %%f2, %s;\n", f32(0)) +
		"\trcp.rn.f32 %f6, %f2;\n" +
		fmt.Sprintf("\tselp.f32 %%f5, %s, %%f6, %%p6;\n", f32(0))

	// log, with the argument floored at 1e-7
	b[13] = fmt.Sprintf("\tmax.f32 %%f6, %%f2, %s;\n", f32(1e-7)) +
		"\tlg2.approx.ftz.f32 %f7, %f6;\n" +
		fmt.Sprintf("\tmul.f32 %%f5, %%f7, %s;\n", f32(ln2))

	// exp, with the argument clamped
	b[14] = clampF("%f6", "%f2", -60, 60) +
		exp("%f5", "%f6")

	b[15] = "\tabs.f32 %f5, %f2;\n"

	// hat: max(0, 1 - |x|)
	b[16] = "\tabs.f32 %f6, %f2;\n" +
		fmt.Sprintf("\tsub.f32 %%f6, %s, %%f6;\n", f32(1)) +
		fmt.Sprintf("\tmax.f32 %%f5, %%f6, %s;\n", f32(0))

	b[17] = "\tmul.f32 %f5, %f2, %f2;\n"
	b[18] = "\tmul.f32 %f6, %f2, %f2;\n\tmul.f32 %f5, %f6, %f2;\n"
	return b
}

// kernelName is the entry point every launch names.
const kernelName = "neat_activate"

// buildPTX assembles the kernel source.
//
// Parameters, in order:
//
//	nets    int32[3*numNets]  per network: first node, node count, first output slot
//	nodes   16-byte records   op (int32), bias (f32), first edge, end edge
//	edges   8-byte records    source node within its network (int32), weight (f32)
//	outIdx  int32[]           node within its network for each output
//	inputs  f32[samples*numIn]
//	values  f32[...]          scratch, nodes-per-net * threads
//	out     f32[numNets*samples*numOut]
//	numNets, samples, numIn, numOut  uint32
func buildPTX() string {
	var b strings.Builder
	b.WriteString(`.version 6.0
.target sm_50
.address_size 64

.visible .entry ` + kernelName + `(
	.param .u64 p_nets,
	.param .u64 p_nodes,
	.param .u64 p_edges,
	.param .u64 p_outidx,
	.param .u64 p_inputs,
	.param .u64 p_values,
	.param .u64 p_out,
	.param .u32 p_num_nets,
	.param .u32 p_samples,
	.param .u32 p_num_in,
	.param .u32 p_num_out
)
{
	.reg .pred %p<8>;
	.reg .b32 %r<40>;
	.reg .f32 %f<40>;
	.reg .b64 %rd<40>;

	ld.param.u64 %rd1, [p_nets];
	ld.param.u64 %rd2, [p_nodes];
	ld.param.u64 %rd3, [p_edges];
	ld.param.u64 %rd4, [p_outidx];
	ld.param.u64 %rd5, [p_inputs];
	ld.param.u64 %rd6, [p_values];
	ld.param.u64 %rd7, [p_out];
	ld.param.u32 %r1, [p_num_nets];
	ld.param.u32 %r2, [p_samples];
	ld.param.u32 %r3, [p_num_in];
	ld.param.u32 %r4, [p_num_out];

	// %r8 = global thread id, %r9 = total threads
	mov.u32 %r5, %ctaid.x;
	mov.u32 %r6, %ntid.x;
	mov.u32 %r7, %tid.x;
	mad.lo.u32 %r8, %r5, %r6, %r7;
	mul.lo.u32 %r9, %r1, %r2;
	setp.ge.u32 %p1, %r8, %r9;
	@%p1 bra DONE;

	// %r10 = network, %r11 = sample
	div.u32 %r10, %r8, %r2;
	rem.u32 %r11, %r8, %r2;

	// The network's record: %r12 first node, %r13 node count, %r14 first output.
	mul.wide.u32 %rd8, %r10, 12;
	add.u64 %rd8, %rd1, %rd8;
	ld.global.s32 %r12, [%rd8];
	ld.global.s32 %r13, [%rd8+4];
	ld.global.s32 %r14, [%rd8+8];

	// %rd9 = this sample's inputs.
	mul.lo.u32 %r15, %r11, %r3;
	mul.wide.u32 %rd9, %r15, 4;
	add.u64 %rd9, %rd5, %rd9;

	// %rd10 = this thread's column of the scratch buffer; %rd11 = the byte
	// distance between one node's row and the next.
	mul.wide.u32 %rd10, %r8, 4;
	add.u64 %rd10, %rd6, %rd10;
	mul.wide.u32 %rd11, %r9, 4;

	mul.wide.s32 %rd12, %r12, 16;
	add.u64 %rd12, %rd2, %rd12;   // node record
	mov.u64 %rd13, %rd10;         // where this node's value goes
	mov.u32 %r16, 0;

NODE_LOOP:
	setp.ge.s32 %p2, %r16, %r13;
	@%p2 bra NODES_DONE;
	ld.global.s32 %r17, [%rd12];
	ld.global.f32 %f1, [%rd12+4];
	ld.global.s32 %r18, [%rd12+8];
	ld.global.s32 %r19, [%rd12+12];
	setp.ge.s32 %p3, %r17, ` + fmt.Sprint(opInput) + `;
	@%p3 bra OP_INPUT;
	setp.eq.s32 %p3, %r17, ` + fmt.Sprint(opBias) + `;
	@%p3 bra OP_BIAS;

	// A computed node: %f2 = bias + sum of weight * source value.
	mov.f32 %f2, %f1;
	mul.wide.s32 %rd14, %r18, 8;
	add.u64 %rd14, %rd3, %rd14;
EDGE_LOOP:
	setp.ge.s32 %p4, %r18, %r19;
	@%p4 bra EDGES_DONE;
	ld.global.s32 %r20, [%rd14];
	ld.global.f32 %f3, [%rd14+4];
	mul.wide.u32 %rd15, %r20, %r9;
	shl.b64 %rd15, %rd15, 2;
	add.u64 %rd15, %rd10, %rd15;
	ld.global.f32 %f4, [%rd15];
	fma.rn.f32 %f2, %f4, %f3, %f2;
	add.s32 %r18, %r18, 1;
	add.u64 %rd14, %rd14, 8;
	bra EDGE_LOOP;
EDGES_DONE:
`)
	// Dispatch on the opcode. Threads of a warp share a network, hence a node
	// table, so they all take the same branch.
	for op := 0; op < len(activationOps); op++ {
		fmt.Fprintf(&b, "\tsetp.eq.s32 %%p5, %%r17, %d;\n\t@%%p5 bra ACT_%d;\n", op, op)
	}
	// An opcode the encoder never emits: pass the sum through.
	b.WriteString("\tmov.f32 %f5, %f2;\n\tbra STORE;\n")
	for op, body := range activationBodies() {
		fmt.Fprintf(&b, "ACT_%d:\n%s\tbra STORE;\n", op, body)
	}
	b.WriteString(`OP_INPUT:
	sub.s32 %r21, %r17, ` + fmt.Sprint(opInput) + `;
	mul.wide.s32 %rd16, %r21, 4;
	add.u64 %rd16, %rd9, %rd16;
	ld.global.f32 %f5, [%rd16];
	bra STORE;
OP_BIAS:
	mov.f32 %f5, ` + f32(1) + `;
STORE:
	st.global.f32 [%rd13], %f5;
	add.u64 %rd13, %rd13, %rd11;
	add.u64 %rd12, %rd12, 16;
	add.s32 %r16, %r16, 1;
	bra NODE_LOOP;

NODES_DONE:
	// out index = (network * samples + sample) * numOut
	mad.lo.u32 %r22, %r10, %r2, %r11;
	mul.lo.u32 %r22, %r22, %r4;
	mov.u32 %r23, 0;
OUT_LOOP:
	setp.ge.u32 %p5, %r23, %r4;
	@%p5 bra DONE;
	add.u32 %r24, %r14, %r23;
	mul.wide.u32 %rd17, %r24, 4;
	add.u64 %rd17, %rd4, %rd17;
	ld.global.s32 %r25, [%rd17];
	mul.wide.u32 %rd18, %r25, %r9;
	shl.b64 %rd18, %rd18, 2;
	add.u64 %rd18, %rd10, %rd18;
	ld.global.f32 %f20, [%rd18];
	add.u32 %r26, %r22, %r23;
	mul.wide.u32 %rd19, %r26, 4;
	add.u64 %rd19, %rd7, %rd19;
	st.global.f32 [%rd19], %f20;
	add.u32 %r23, %r23, 1;
	bra OUT_LOOP;
DONE:
	ret;
}
`)
	return b.String()
}
