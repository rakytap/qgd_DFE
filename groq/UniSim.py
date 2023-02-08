import numpy as np
from numba import njit
from numba.np.unsafe.ndarray import to_fixed_tuple
import random
from functools import lru_cache
from qgd_python.decomposition.qgd_N_Qubit_Decomposition_adaptive import qgd_N_Qubit_Decomposition_adaptive       
# number of adaptive levels
levels = 5

# set true to limit calculations to real numbers
real=False

isGroq = True

def to_cirq(unitary, qbit_num, parameters, target_qbits, control_qbits):
    import cirq
    from cirq import Gate
    class U3Gate(cirq.Gate):
        def __init__(self, params):
            super(U3Gate, self)
            self.params = params        
        def _num_qubits_(self):
            return 1        
        def _unitary_(self):
            return make_u3(self.params)        
        def _circuit_diagram_info_(self, args):
            return "U"
    circuit = cirq.Circuit()
    q = cirq.LineQubit.range(qbit_num)
    for param, target_qbit, control_qbit in zip(parameters, target_qbits, control_qbits):
        if control_qbit is None or target_qbit == control_qbit:
            circuit.append(U3Gate(param).on(q[target_qbit]))
        else:
            circuit.append(cirq.ry(param[0]*2).on(q[target_qbit]).controlled_by(q[control_qbit])) 
    return circuit
def cirq_us(circuit):
    import cirq
    #from cirq.contrib.qasm_import import circuit_from_qasm
    #circuit = circuit_from_qasm(qasm)
    #simulator = cirq.Simulator()
    #result = simulator.run(circuit, repetitions=1)
    Umtx = cirq.unitary(circuit)
    return Umtx
def to_qulacs(unitary, qbit_num, parameters, target_qbits, control_qbits):
    from qulacs import QuantumCircuit
    from qulacs.gate import RY, U3, DenseMatrix, to_matrix_gate
    circuit = QuantumCircuit(qbit_num)
    for param, target_qbit, control_qbit in zip(parameters, target_qbits, control_qbits):
        if control_qbit is None or target_qbit == control_qbit:
            circuit.add_gate(U3(target_qbit, param[0]*2, param[1], param[2])) # DenseMatrix(target_qbit, make_u3(param)))
        else:
            gate = to_matrix_gate(RY(target_qbit, -param[0]*2))
            gate.add_control_qubit(control_qbit, 1)
            circuit.add_gate(gate)
    return circuit
def qulacs_us(circuit):
    from qulacs.circuit import QuantumCircuitOptimizer
    return QuantumCircuitOptimizer().merge_all(circuit).get_matrix()
def to_qibo(unitary, qbit_num, parameters, target_qbits, control_qbits):
    from qibo import gates, models
    circuit = models.Circuit(qbit_num)
    for param, target_qbit, control_qbit in zip(parameters, target_qbits, control_qbits):
        if control_qbit is None or target_qbit == control_qbit:
            circuit.add(gates.Unitary(make_u3(param), target_qbit)) #gates.U3(target_qbit, param[0]*2, param[1], param[2])
        else:
            circuit.add(gates.CRY(control_qbit, target_qbit, param[0]*2))
    return circuit
def qibo_us(circuit):
    return circuit.unitary()
def qiskit_us(circuit, usefloat=False):
    from qiskit import Aer
    from qiskit import execute
    backend = Aer.get_backend('unitary_simulator')
    if usefloat: backend.set_option("precision", "single")
    job = execute(circuit, backend)
    result=job.result()
    U3_qiskit = result.get_unitary(circuit)
    U3_qiskit = np.asarray(U3_qiskit)
    return U3_qiskit
def to_qiskit(unitary, qbit_num, parameters, target_qbits, control_qbits):
    from qiskit import QuantumCircuit
    circuit = QuantumCircuit(qbit_num)
    circuit.unitary(unitary, [i for i in range(qbit_num)])
    for param, target_qbit, control_qbit in zip(parameters, target_qbits, control_qbits):
        if control_qbit is None or target_qbit == control_qbit:
            circuit.u(param[0]*2, param[1], param[2], target_qbit)
        else:
            circuit.cry(param[0]*2, control_qbit, target_qbit)
    return circuit
def qiskit_oracle(unitary, qbit_num, parameters, target_qbits, control_qbits, usefloat=False):
    return qiskit_us(to_qiskit(unitary, qbit_num, parameters, target_qbits, control_qbits), usefloat)
@njit
def make_u3(parameters):
    return np.array(
        [[np.cos(parameters[0]*2/2), -np.exp(parameters[2]*1j)*np.sin(parameters[0]*2/2)],
         [np.exp(parameters[1]*1j)*np.sin(parameters[0]*2/2), np.exp((parameters[1]+parameters[2])*1j)*np.cos(parameters[0]*2/2)]])
@njit
def make_ry(parameters):
    return make_u3([parameters[0], 0, 0])
    #return np.array(
    #    [[np.cos(parameters[0]*2/2), -np.sin(parameters[0]*2/2)],
    #     [np.sin(parameters[0]*2/2), np.cos(parameters[0]*2/2)]])
@njit
def make_controlled(gate):
    return np.block([[np.eye(2), np.zeros((2, 2))], [np.zeros((2, 2)), gate]]) #[np.ix_(*([[0,2,1,3]]*2))]
@njit
def make_cry(parameters):
    return make_ry(parameters) #make_controlled(make_ry(parameters))
@njit
def twoByTwoFloat(A, B):
    res = np.empty(B.shape, dtype=B.dtype)
    for j in range(2):
        for i in range(B.shape[1]):
            res[j,i] = (np.real(A[j,0])*np.real(B[0,i])-np.imag(A[j,0])*np.imag(B[0,i])) + (np.real(A[j,1])*np.real(B[1,i])-np.imag(A[j,1])*np.imag(B[1,i]))
            res[j,i] += ((np.real(A[j,0])*np.imag(B[0,i])+np.imag(A[j,0])*np.real(B[0,i])) + (np.real(A[j,1])*np.imag(B[1,i])+np.imag(A[j,1])*np.real(B[1,i]))) * 1j
            #((np.real(A[j,0])*np.imag(B[0,i])+np.real(A[j,1])*np.imag(B[1,i])) + (np.imag(A[j,0])*np.real(B[0,i])+np.imag(A[j,1])*np.real(B[1,i]))) * 1j
    return res
#@njit
def apply_to_qbit(unitary, num_qbits, target_qbit, control_qbit, gate):
    pow2qb = 1 << num_qbits
    t = np.arange(num_qbits)
    if not control_qbit is None:
        t[:-1] = np.roll(t[:-1], (target_qbit - control_qbit) % num_qbits)
        gate = make_controlled(gate)
    t = np.roll(t, -target_qbit)
    idxs = np.arange(pow2qb).reshape(*((2,)*num_qbits)).transpose(t).flatten().tolist()
    return np.kron(np.eye(pow2qb>>(1 if control_qbit is None else 2), dtype=np.bool_), gate)[np.ix_(idxs, idxs)].astype(unitary.dtype) @ unitary
@lru_cache(128)
def make_apply_to_qbit_loop(num_qbits):
    twos = tuple([2]*num_qbits)
    @njit
    def apply_to_qbit_loop(unitary, _, target_qbit, control_qbit, gate):
        pow2qb = 1 << num_qbits
        t = np.roll(np.arange(num_qbits), target_qbit)
        idxs = np.arange(pow2qb).reshape(twos).transpose(to_fixed_tuple(t, num_qbits)).copy().reshape(-1, 2) #.reshape(*([2]*num_qbits)).transpose(t).reshape(-1, 2)
        for pair in (idxs if control_qbit is None else idxs[(idxs[:,0] & (1<<control_qbit)) != 0,:]):
            unitary[pair,:] = twoByTwoFloat(gate, unitary[pair,:])
            #unitary[pair,:] = gate @ unitary[pair,:]
        return unitary
    return apply_to_qbit_loop
def process_gates32(unitary, num_qbits, parameters, target_qbits, control_qbits):
    return process_gates(unitary.astype(np.complex64), num_qbits, parameters, target_qbits, control_qbits).astype(np.complex128)
def process_gates(unitary, num_qbits, parameters, target_qbits, control_qbits):
    if unitary.dtype == np.dtype(np.complex128): unitary = np.copy(unitary)
    return process_gates_loop(unitary, num_qbits, parameters, target_qbits, control_qbits, make_apply_to_qbit_loop(num_qbits)) #apply_to_qbit
@njit
def process_gates_loop(unitary, num_qbits, parameters, target_qbits, control_qbits, apply_to_qbit_func):
    for param, target_qbit, control_qbit in zip(parameters, target_qbits, control_qbits):
        unitary = apply_to_qbit_func(unitary, num_qbits, target_qbit, None if control_qbit == target_qbit else control_qbit, (make_u3(param) if control_qbit is None or control_qbit==target_qbit else make_cry(param)).astype(unitary.dtype))
    return unitary
def trace_corrections(result, num_qbits):
    import math
    def mathcomb(n, k): return math.factorial(n) // (math.factorial(k)*math.factorial(n-k)) 
    pow2qb = 1 << num_qbits
    return np.array([np.trace(np.real(result)),
        np.sum(np.real(result[[c ^ (1<<i) for i in range(num_qbits) for c in range(pow2qb)], list(range(pow2qb))*num_qbits])),
        np.sum(np.real(result[[c ^ ((1<<i)+(1<<j)) for i in range(num_qbits-1) for j in range(i+1, num_qbits) for c in range(pow2qb)], list(range(pow2qb))*mathcomb(num_qbits, 2)]))])
def costfunc(traces, pow2qb):
    return 1 - (traces[0] + np.sqrt(0)*(traces[1]/1.7 + traces[2]/2.0)) / pow2qb 
def get_gate_structure(levels, qbit_num):
    target_qbits, control_qbits = [], []
    for _ in range(levels):        
        #adaptive blocks
        for target_qubit in range(qbit_num):
            for control_qubit in range(target_qubit+1, qbit_num, 1):
                target_qbits.extend([target_qubit, control_qubit, target_qubit])
                control_qbits.extend([target_qubit, control_qubit, control_qubit])
    # U3 gates
    for target_qubit in range(qbit_num):
        target_qbits.append(target_qubit)
        control_qbits.append(target_qubit)
    return np.array(target_qbits, dtype=np.uint8), np.array(control_qbits, dtype=np.uint8)

##
# @brief Call to construct random parameter, with limited number of non-trivial adaptive layers
# @param num_of_parameters The number of parameters
def create_randomized_parameters( qbit_num, num_of_parameters, real=False ):


    parameters = np.zeros(num_of_parameters)

    # the number of adaptive layers in one level
    num_of_adaptive_layers = int(qbit_num*(qbit_num-1)/2 * levels)
    
    if (real):
        
        for idx in range(qbit_num):
            parameters[idx*3] = np.random.rand(1)*2*np.pi

    else:
        parameters[0:3*qbit_num] = np.random.rand(3*qbit_num)*np.pi
        pass

    nontrivial_adaptive_layers = np.zeros( (num_of_adaptive_layers ))
    
    for layer_idx in range(num_of_adaptive_layers) :

        nontrivial_adaptive_layer = random.randint(0,1)
        nontrivial_adaptive_layers[layer_idx] = nontrivial_adaptive_layer

        if (nontrivial_adaptive_layer) :
        
            # set the random parameters of the chosen adaptive layer
            start_idx = qbit_num*3 + layer_idx*7
            
            if (real):
                parameters[start_idx]   = np.random.rand(1)*2*np.pi
                parameters[start_idx+1] = np.random.rand(1)*2*np.pi
                parameters[start_idx+4] = np.random.rand(1)*2*np.pi
            else:
                end_idx = start_idx + 7
                parameters[start_idx:end_idx] = np.random.rand(7)*2*np.pi
         
        
    
    #print( parameters )
    return parameters, nontrivial_adaptive_layers

def perf_collection():
    import timeit, os, pickle
    params = {}
    if os.path.exists("squanderperf.pickle"):
        with open("squanderperf.pickle", 'rb') as f:
            results = pickle.load(f)
    else: results = {}
    if isGroq:
        import groq.runtime as runtime
        driver = runtime.Driver()
        alldev = list(range(1, len(driver.devices)+1))
    else: alldev = (0, 1, 2, 3)    
    qbit_range = list(range(5, 10+1))
    if not isGroq in results: results[isGroq] = {}
    for numdev in alldev:
        results[isGroq][numdev] = {}
        for qbit_num in qbit_range:
            matrix_size = 1 << qbit_num
            # creating a class to decompose the unitary
            result = [None]
            def initInvoke():
                result[0] = qgd_N_Qubit_Decomposition_adaptive( np.eye(matrix_size), level_limit_max=levels, level_limit_min=0, accelerator_num = numdev )
            inittime = timeit.timeit(initInvoke, number=1)
            cDecompose = result[0]
            def uploadInvoke():
                cDecompose.Upload_Umtx_to_DFE() #program and matrix load are currently both here
            uptime = timeit.timeit(uploadInvoke, number=1)
            uptime = timeit.timeit(uploadInvoke, number=1)
            # adding decomposing layers to the gate structure
            for idx in range(levels):
                cDecompose.add_Adaptive_Layers()
            
            cDecompose.add_Finalyzing_Layer_To_Gate_Structure()
            cDecompose.set_Cost_Function_Variant(0)
            
            # get the number of free parameters
            num_of_parameters = cDecompose.get_Parameter_Num()
            print("Qubits:", qbit_num, "Levels:", levels, "Parameters:", num_of_parameters) 
            # create randomized parameters            
            if numdev == 0 or isGroq and numdev == 1:
                parameters, nontrivial_adaptive_layers = create_randomized_parameters( qbit_num, num_of_parameters, real=real )
                params[qbit_num] = parameters
            else: parameters = params[qbit_num]
            result = [None]
            def dfeInvoke():
                result[0] = cDecompose.Optimization_Problem_Combined( parameters )
            t = timeit.timeit(dfeInvoke, number=1)
            print(numdev, qbit_num, inittime, uptime, t, result[0][0])
            results[isGroq][numdev][qbit_num] = (uptime, t)
            if numdev == 0:
                if not -1 in results[isGroq]: results[isGroq][-1] = {}
                cDecompose.set_Optimized_Parameters( parameters )
                cDecompose.Prepare_Gates_To_Export()
                target_qbits, control_qbits = get_gate_structure(levels, qbit_num)
                i, p = 0, []
                for t, c in zip(reversed(target_qbits), reversed(control_qbits)):
                    if t == c: p.append(parameters[i:i+3]); i += 3
                    else: p.append([parameters[i], 0, 0]); i += 1
                p = np.array(list(reversed(p)), dtype=np.float64)
                #gates = cDecompose.get_Gates()
                #target_qbitsC = np.array([x['target_qbit'] for x in reversed(gates)], dtype=np.uint8)
                #control_qbitsC = np.array([x['control_qbit'] if 'control_qbit' in x else x['target_qbit'] for x in reversed(gates)], dtype=np.uint8)
                #pC = np.array([[x['Theta'], x['Phi'] if 'Phi' in x else 0.0, x['Lambda'] if 'Lambda' in x else 0.0] for x in reversed(gates)], dtype=np.float64)
                #circuit = to_qiskit(np.eye(matrix_size, dtype=np.complex128), qbit_num, p, target_qbits, control_qbits)
                #qasm = cDecompose.get_Quantum_Circuit().qasm()
                def py():
                    result[0] = costfunc(trace_corrections(process_gates(np.eye(matrix_size, dtype=np.complex128), qbit_num, p, target_qbits, control_qbits), qbit_num), matrix_size)
                def qiskit():
                    #result[0] = costfunc(trace_corrections(qiskit_us(cDecompose.get_Quantum_Circuit()), qbit_num), matrix_size)
                    result[0] = costfunc(trace_corrections(qiskit_oracle(np.eye(matrix_size, dtype=np.complex128), qbit_num, p, target_qbits, control_qbits), qbit_num), matrix_size)
                def cirq():
                    result[0] = costfunc(trace_corrections(cirq_us(to_cirq(np.eye(matrix_size, dtype=np.complex128), qbit_num, p, target_qbits, control_qbits)), qbit_num), matrix_size)
                def qibo():
                    result[0] = costfunc(trace_corrections(qibo_us(to_qibo(np.eye(matrix_size, dtype=np.complex128), qbit_num, p, target_qbits, control_qbits)), qbit_num), matrix_size)
                def qulacs():
                    result[0] = costfunc(trace_corrections(qulacs_us(to_qulacs(np.eye(matrix_size, dtype=np.complex128), qbit_num, p, target_qbits, control_qbits)), qbit_num), matrix_size)
                for func in (qulacs,): #(py, qiskit, cirq, qibo, qulacs):
                    if any(results[isGroq][-1][x][1] > 45 for x in results[isGroq][-1] if x < qbit_num): continue
                    continue
                    t = timeit.timeit(func, number=1+num_of_parameters)
                    print(t, result[0])
                    results[isGroq][-1][qbit_num] = (0, t)
    with open("squanderperf.pickle", 'wb') as f:
        pickle.dump(results, f)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(8.5, 4)
    ax[0].set_xticks(qbit_range)
    ax[1].set_xticks(qbit_range)
    ax[0].set_xlabel("# of qubits")
    ax[1].set_xlabel("# of qubits")
    ax[0].set_ylabel("Time (s)")
    ax[1].set_ylabel("Time (s)")
    ax[1].set_yscale('log', base=2)
    ax[0].set_title("Quantum Unitary Simulator Initialization")
    ax[1].set_title("Quantum Unitary Simulator Performance")
    import itertools
    marker1 = itertools.cycle(['o', '*', 'x', '+', 's', 'p', '1', '2', '3', '4', '8', 'P', 'h'])
    marker2 = itertools.cycle(['o', '*', 'x', '+', 's', 'p', '1', '2', '3', '4', '8', 'P', 'h'])
    print(["Qulacs" if numdev==-1 else ("CPU AVX" if numdev==0 else str(numdev) + " " + ("Groq" if g else "DFE")) + " " + str(x) + " qubits " + str(results[g][numdev][x][1]) for g in results for numdev in results[g] for x in sorted(results[g][numdev])])
    for g in results:
        for numdev in results[g]:
            r = list(sorted(results[g][numdev].keys()))
            ax[0].plot(r, [results[g][numdev][x][0] for x in r], label="Qulacs" if numdev==-1 else ("CPU AVX" if numdev==0 else str(numdev) + " " + ("Groq" if g else "DFE")), linewidth=1, marker=next(marker1))
            ax[1].plot(r, [results[g][numdev][x][1] for x in r], label="Qulacs" if numdev==-1 else ("CPU AVX" if numdev==0 else str(numdev) + " " + ("Groq" if g else "DFE")), linewidth=1, marker=next(marker2))
    ax[0].legend()
    ax[1].legend()
    plt.rcParams.update({'font.size': 12})
    fig.savefig('squanderperf.svg', format='svg')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(8.5, 3)
    ax.set_xticks(qbit_range)
    ax.set_xlabel("# of qubits")
    ax.set_ylabel("Time (s)")
    ax.set_yscale('log', base=2)
    #ax.set_title("Quantum Unitary Simulator Performance")
    marker = itertools.cycle(['o', '*', 'x', '+', 's', 'p', '1', '2', '3', '4', '8', 'P', 'h'])
    for g in results:
        for numdev in results[g]:
            if numdev >= 2: continue
            r = list(sorted(results[g][numdev].keys()))
            ax.plot(r, [results[g][numdev][x][1] for x in r], label="Qulacs" if numdev==-1 else ("SQUANDER CPU" if numdev==0 else ("SQUANDER TSP" if g else "SQUANDER FPGA")), linewidth=1, marker=next(marker))
    ax.legend()
    fig.savefig('squanderperfpaper.svg', format='svg')    
perf_collection()
