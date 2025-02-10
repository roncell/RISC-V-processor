import os
import argparse

# Memory size definition. The real-world memory size should be 2^32.
# For this assignment, due to space constraints, we use a smaller number.
# Note that memory is still 32-bit addressable.
MemSize = 1000


class InsMem(object):  # Handles instruction memory operations
    def __init__(self, name, ioDir):
        self.id = name

        with open(ioDir + os.sep + "imem.txt") as im:
            # Load instruction memory from the provided file
            self.IMem = [data.replace("\n", "") for data in im.readlines()]

    def readInstr(self, ReadAddress):
        # Reads instruction memory and returns a 32-bit hexadecimal value.
        # Converts binary data to a decimal number, then formats as hex.
        inst = int("".join(self.IMem[ReadAddress: ReadAddress + 4]), 2)
        return format(inst, '#010x')  # Output as 0x followed by 8 hex digits

    def read_instr(self, read_address: int) -> str:
        # Reads 32-bit binary instruction from memory as a string.
        return "".join(self.IMem[read_address: read_address + 4])


class DataMem(object):  # Handles data memory operations
    def __init__(self, name, ioDir):
        self.id = name
        self.ioDir = ioDir
        with open(ioDir + os.sep + "dmem.txt") as dm:
            # Load data memory from the provided file
            self.DMem = [data.replace("\n", "") for data in dm.readlines()]
        # Extend the memory with zeros to the predefined size
        self.DMem = self.DMem + (['00000000'] * (MemSize - len(self.DMem)))

    def readDataMem(self, ReadAddress):
        # Reads 32-bit data from memory and returns it as a hexadecimal value.
        data32 = int("".join(self.DMem[ReadAddress: ReadAddress + 4]), 2)
        return format(data32, '#010x')  # Output as 0x followed by 8 hex digits

    def writeDataMem(self, Address, WriteData):
        # Writes data into the byte-addressable memory
        mask8 = int('0b11111111', 2)  # 8-bit mask
        data8_arr = []

        for j in range(4):
            # Extract 8 bits at a time from the 32-bit data
            data8_arr.append(WriteData & mask8)
            WriteData = WriteData >> 8

        for i in range(4):
            # Store most significant byte at the smallest address
            self.DMem[Address + i] = format(data8_arr.pop(), '08b')

    def read_data_mem(self, read_addr: str) -> str:
        # Reads 32-bit data as a binary string from memory
        read_addr_int = bin2int(read_addr)
        return "".join(self.DMem[read_addr_int: read_addr_int + 4])

    def write_data_mem(self, addr: str, write_data: str):
        # Writes binary string data to memory at the specified address
        addr_int = bin2int(addr)
        for i in range(4):
            self.DMem[addr_int + i] = write_data[8 * i: 8 * (i + 1)]

    def outputDataMem(self):
        # Outputs the current state of data memory to a file
        resPath = self.ioDir + os.sep + self.id + "_DMEMResult.txt"
        with open(resPath, "w") as rp:
            rp.writelines([str(data) + "\n" for data in self.DMem])


class RegisterFile(object):  # Handles CPU register operations
    def __init__(self, ioDir):
        self.outputFile = ioDir + "RFResult.txt"
        # Initialize 32 registers, each 32-bit wide
        self.Registers = [0x0 for i in range(32)]
        self.registers = [int2bin(0)
                          for _ in range(32)]  # For multi-stage cores

    def readRF(self, Reg_addr):  # Reads data from a specific register
        return self.Registers[Reg_addr]

    def writeRF(self, Reg_addr, Wrt_reg_data):  # Writes data into a register
        if Reg_addr != 0:  # Register 0 is read-only and always 0
            self.Registers[Reg_addr] = Wrt_reg_data & (
                (1 << 32) - 1)  # Ensure data fits into 32 bits

    def outputRF(self, cycle):
        # Outputs the current state of registers to a file
        op = ["State of RF after executing cycle:  " +
              str(cycle) + "\n"]
        op.extend([format(val, '032b')+"\n" for val in self.Registers])
        perm = "w" if cycle == 0 else "a"
        with open(self.outputFile, perm) as file:
            file.writelines(op)

    # Multi-stage pipeline operations
    def read_RF(self, reg_addr: str) -> str:
        return self.registers[bin2int(reg_addr)]

    def write_RF(self, reg_addr: str, wrt_reg_data: str):
        if reg_addr != "00000":  # Register 0 remains read-only
            self.registers[bin2int(reg_addr)] = wrt_reg_data

    def output_RF(self, cycle):
        op = ["State of RF after executing cycle:" + str(cycle) + "\n"]
        op.extend([f"{val}" + "\n" for val in self.registers])
        perm = "w" if cycle == 0 else "a"
        with open(self.outputFile, perm) as file:
            file.writelines(op)


class State(object):  # Represents the state of the pipeline
    def __init__(self):
        self.IF = {"nop": bool(False), "PC": int(0), "taken": bool(False)}
        self.ID = {"nop": bool(False), "instr": str(
            "0"*32), "PC": int(0), "hazard_nop": bool(False)}
        self.EX = {"nop": bool(False), "instr": str("0"*32), "Read_data1": str("0"*32), "Read_data2": str("0"*32), "Imm": str("0"*32), "Rs": str("0"*5), "Rt": str("0"*5), "Wrt_reg_addr": str("0"*5), "is_I_type": bool(False), "rd_mem": bool(False),
                   # alu_op: 00 -> add, 01 -> and, 10 -> or, 11 -> xor
                   "wrt_mem": bool(False), "alu_op": str("00"), "wrt_enable": bool(False)}
        self.MEM = {"nop": bool(False), "ALUresult": str("0"*32), "Store_data": str("0"*32), "Rs": str("0"*5), "Rt": str("0"*5), "Wrt_reg_addr": str("0"*5), "rd_mem": bool(False),
                    "wrt_mem": bool(False), "wrt_enable": bool(False)}
        self.WB = {"nop": bool(False), "Wrt_data": str("0"*32), "Rs": str(
            "0"*5), "Rt": str("0"*5), "Wrt_reg_addr": str("0"*5), "wrt_enable": bool(False)}


class Core(object):  # Represents the CPU core
    def __init__(self, ioDir, imem, dmem):
        self.myRF = RegisterFile(ioDir)
        self.cycle = 0
        self.inst = 0
        self.halted = False
        self.ioDir = ioDir
        self.state = State()
        self.nextState = State()
        self.ext_imem = imem
        self.ext_dmem = dmem

# ------------------------------------
# Single-cycle core implementation

# ALU Arithmetic Operations


def Calculate_R(funct7, funct3, rs1, rs2):
    rd = 0
    # Perform addition if funct7 and funct3 indicate ADD operation
    if funct7 == 0b0000000 and funct3 == 0b000:
        rd = rs1 + rs2

    # Perform subtraction if funct7 and funct3 indicate SUB operation
    elif funct7 == 0b0100000 and funct3 == 0b000:
        rd = rs1 - rs2

    # Perform bitwise XOR if funct7 and funct3 indicate XOR operation
    elif funct7 == 0b0000000 and funct3 == 0b100:
        rd = rs1 ^ rs2

    # Perform bitwise OR if funct7 and funct3 indicate OR operation
    elif funct7 == 0b0000000 and funct3 == 0b110:
        rd = rs1 | rs2

    # Perform bitwise AND if funct7 and funct3 indicate AND operation
    elif funct7 == 0b0000000 and funct3 == 0b111:
        rd = rs1 & rs2

    return rd

# Compute the sign-extended value from a binary input


def sign_extend(val, sign_bit):
    # Check if the sign bit is set
    if (val & (1 << sign_bit)) != 0:
        # Convert to a negative value using two's complement
        val = val - (1 << (sign_bit + 1))
    return val

# ALU Immediate Operations


def Calculate_I(funct3, rs1, imm):
    rd = 0
    # Perform ADDI operation
    if funct3 == 0b000:
        rd = rs1 + sign_extend(imm, 11)

    # Perform XORI operation
    elif funct3 == 0b100:
        rd = rs1 ^ sign_extend(imm, 11)

    # Perform ORI operation
    elif funct3 == 0b110:
        rd = rs1 | sign_extend(imm, 11)

    # Perform ANDI operation
    elif funct3 == 0b111:
        rd = rs1 & sign_extend(imm, 11)

    return rd

# Single-stage CPU implementation


class SingleStageCore(Core):
    def __init__(self, ioDir, imem, dmem):
        super(SingleStageCore, self).__init__(
            ioDir + os.sep + "SS_", imem, dmem)
        self.opFilePath = ioDir + os.sep + "StateResult_SS.txt"

    def step(self):
        # Fetch instruction from memory and decode its opcode
        fetchedInstr = int(self.ext_imem.readInstr(
            self.state.IF["PC"]), 16)  # Convert hexadecimal to integer
        # Extract the least significant 7 bits
        opcode = fetchedInstr & (2 ** 7 - 1)

        # Decode the instruction and execute the corresponding operation
        self.Decode(opcode, fetchedInstr)

        # Check for halt condition
        self.halted = False
        if self.state.IF["nop"]:
            self.halted = True

        # Update the program counter for the next instruction
        if not self.state.IF["taken"] and self.state.IF["PC"] + 4 < len(self.ext_imem.IMem):
            self.nextState.IF["PC"] = self.state.IF["PC"] + 4
        else:
            self.state.IF["taken"] = False  # Reset taken flag after a branch

        # Output the state of registers after the current cycle
        self.myRF.outputRF(self.cycle)

        # Log the state after the current cycle
        self.printState(self.nextState, self.cycle)

        # End of cycle: Update the current state with next state values
        self.state = self.nextState
        self.cycle += 1
        self.inst += 1  # Increment instruction counter

    def Decode(self, opcode, fetchedInstr):
        # Handle R-type instructions
        if opcode == 0b0110011:
            # Decode individual instruction fields
            funct7 = fetchedInstr >> 25
            funct3 = (fetchedInstr >> 12) & ((1 << 3) - 1)
            rs2 = (fetchedInstr >> 20) & ((1 << 5) - 1)
            rs1 = (fetchedInstr >> 15) & ((1 << 5) - 1)
            rd = (fetchedInstr >> 7) & ((1 << 5) - 1)

            # Fetch register data and calculate the result
            data_rs1 = self.myRF.readRF(rs1)
            data_rs2 = self.myRF.readRF(rs2)
            data_rd = Calculate_R(funct7, funct3, data_rs1, data_rs2)

            # Write the result back to the destination register
            self.myRF.writeRF(rd, data_rd)

        # Handle I-type instructions
        elif opcode == 0b0010011:
            imm = fetchedInstr >> 20 & ((1 << 12) - 1)
            funct3 = (fetchedInstr >> 12) & ((1 << 3) - 1)
            rs1 = (fetchedInstr >> 15) & ((1 << 5) - 1)
            rd = (fetchedInstr >> 7) & ((1 << 5) - 1)

            data_rs1 = self.myRF.readRF(rs1)
            data_rd = Calculate_I(funct3, data_rs1, imm)

            self.myRF.writeRF(rd, data_rd)

        # Handle JAL (jump and link) instruction
        elif opcode == 0b1101111:
            imm19_12 = (fetchedInstr >> 12) & ((1 << 8) - 1)
            imm11 = (fetchedInstr >> 20) & 1
            imm10_1 = (fetchedInstr >> 21) & ((1 << 10) - 1)
            imm20 = (fetchedInstr >> 31) & 1
            imm = (imm20 << 20) | (imm10_1 << 1) | (
                imm11 << 11) | (imm19_12 << 12)

            rd = (fetchedInstr >> 7) & ((1 << 5) - 1)

            self.myRF.writeRF(rd, self.state.IF["PC"] + 4)
            self.nextState.IF["PC"] = self.state.IF["PC"] + \
                sign_extend(imm, 20)
            self.state.IF["taken"] = True

        # Handle B-type instructions (conditional branches)
        elif opcode == 0b1100011:
            imm11 = (fetchedInstr >> 7) & 1
            imm4_1 = (fetchedInstr >> 8) & ((1 << 4) - 1)
            imm10_5 = (fetchedInstr >> 25) & ((1 << 6) - 1)
            imm12 = (fetchedInstr >> 31) & 1
            imm = (imm11 << 11) | (imm4_1 << 1) | (
                imm10_5 << 5) | (imm12 << 12)

            rs2 = (fetchedInstr >> 20) & ((1 << 5) - 1)
            rs1 = (fetchedInstr >> 15) & ((1 << 5) - 1)
            funct3 = (fetchedInstr >> 12) & ((1 << 3) - 1)

            data_rs1 = self.myRF.readRF(rs1)
            data_rs2 = self.myRF.readRF(rs2)

            # BEQ and BNE conditions
            if funct3 == 0b000 and data_rs1 == data_rs2:
                self.nextState.IF["PC"] = self.state.IF["PC"] + \
                    sign_extend(imm, 12)
                self.state.IF["taken"] = True
            elif funct3 != 0b000 and data_rs1 != data_rs2:
                self.nextState.IF["PC"] = self.state.IF["PC"] + \
                    sign_extend(imm, 12)
                self.state.IF["taken"] = True

        # Handle LW (load word) instruction
        elif opcode == 0b0000011:
            imm = fetchedInstr >> 20
            rs1 = (fetchedInstr >> 15) & ((1 << 5) - 1)
            rd = (fetchedInstr >> 7) & ((1 << 5) - 1)

            self.myRF.writeRF(
                rd,
                int(self.ext_dmem.readDataMem(
                    self.myRF.readRF(rs1) + sign_extend(imm, 11)
                ), 16)
            )

        # Handle SW (store word) instruction
        elif opcode == 0b0100011:
            imm11_5 = fetchedInstr >> 25
            imm4_0 = (fetchedInstr >> 7) & ((1 << 5) - 1)
            imm = (imm11_5 << 5) | imm4_0

            rs1 = (fetchedInstr >> 15) & ((1 << 5) - 1)
            rs2 = (fetchedInstr >> 20) & ((1 << 5) - 1)

            self.ext_dmem.writeDataMem(
                (self.myRF.readRF(rs1) + sign_extend(imm, 11)) & ((1 << 32) - 1),
                self.myRF.readRF(rs2)
            )

        # Handle HALT instruction
        else:
            self.state.IF["nop"] = True

    # Log the state after each cycle into the output file
    def printState(self, state, cycle):
        printstate = [
            "State after executing cycle: " + str(cycle) + "\n",
            "IF.PC: " + str(state.IF["PC"]) + "\n",
            "IF.nop: " + str(state.IF["nop"]) + "\n"
        ]
        perm = "w" if cycle == 0 else "a"
        with open(self.opFilePath, perm) as wf:
            wf.writelines(printstate)

# -----------------------------------------
# Five-stage pipeline implementation

# Instruction Fetch Stage State


class InstructionFetchState:
    def __init__(self) -> None:
        self.nop: bool = False  # No operation flag
        self.PC: int = 0  # Program Counter

    def __dict__(self):
        # Dictionary representation of the state
        return {"PC": self.PC, "nop": self.nop}


# Instruction Decode Stage State
class InstructionDecodeState:
    def __init__(self) -> None:
        self.nop: bool = True  # No operation flag
        self.hazard_nop: bool = False  # Hazard handling flag
        self.PC: int = 0  # Program Counter
        self.instr: str = "0" * 32  # 32-bit instruction as a string

    def __dict__(self):
        # Dictionary representation of the state
        return {"Instr": self.instr[::-1], "nop": self.nop}


# Execution Stage State
class ExecutionState:
    def __init__(self) -> None:
        self.nop: bool = True  # No operation flag
        self.instr: str = ""  # Current instruction
        self.read_data_1: str = "0" * 32  # First operand
        self.read_data_2: str = "0" * 32  # Second operand
        self.imm: str = "0" * 32  # Immediate value
        self.rs: str = "0" * 5  # Source register 1
        self.rt: str = "0" * 5  # Source register 2
        self.write_reg_addr: str = "0" * 5  # Destination register
        self.is_I_type: bool = False  # Flag for I-type instruction
        self.read_mem: bool = False  # Memory read flag
        self.write_mem: bool = False  # Memory write flag
        # ALU operation (00 -> add, 01 -> and, 10 -> or, 11 -> xor)
        self.alu_op: str = "00"
        self.write_enable: bool = False  # Register write enable flag

    def __dict__(self):
        # Dictionary representation of the state
        return {
            "nop": self.nop,
            "instr": self.instr[::-1],
            "Operand1": self.read_data_1,
            "Operand2": self.read_data_2,
            "Imm": self.imm,
            "Rs": self.rs,
            "Rt": self.rt,
            "Wrt_reg_addr": self.write_reg_addr,
            "is_I_type": int(self.is_I_type),
            "rd_mem": int(self.read_mem),
            "wrt_mem": int(self.write_mem),
            "alu_op": "".join(list(map(str, self.alu_op))),
            "wrt_enable": int(self.write_enable),
        }


# Memory Access Stage State
class MemoryAccessState:
    def __init__(self) -> None:
        self.nop: bool = True  # No operation flag
        self.alu_result: str = "0" * 32  # Result from ALU
        self.store_data: str = "0" * 32  # Data to store in memory
        self.rs: str = "0" * 5  # Source register 1
        self.rt: str = "0" * 5  # Source register 2
        self.write_reg_addr: str = "0" * 5  # Destination register
        self.read_mem: bool = False  # Memory read flag
        self.write_mem: bool = False  # Memory write flag
        self.write_enable: bool = False  # Register write enable flag

    def __dict__(self):
        # Dictionary representation of the state
        return {
            "nop": self.nop,
            "ALUresult": self.alu_result,
            "Store_data": self.store_data,
            "Rs": self.rs,
            "Rt": self.rt,
            "Wrt_reg_addr": self.write_reg_addr,
            "rd_mem": int(self.read_mem),
            "wrt_mem": int(self.write_mem),
            "wrt_enable": int(self.write_enable),
        }


# Write Back Stage State
class WriteBackState:
    def __init__(self) -> None:
        self.nop: bool = True  # No operation flag
        self.write_data: str = "0" * 32  # Data to write back to the register
        self.rs: str = "0" * 5  # Source register 1
        self.rt: str = "0" * 5  # Source register 2
        self.write_reg_addr: str = "0" * 5  # Destination register
        self.write_enable: bool = False  # Register write enable flag

    def __dict__(self):
        # Dictionary representation of the state
        return {
            "nop": self.nop,
            "Wrt_data": self.write_data,
            "Rs": self.rs,
            "Rt": self.rt,
            "Wrt_reg_addr": self.write_reg_addr,
            "wrt_enable": int(self.write_enable),
        }


# Full State Representation for Five Stages
class State_five(object):
    def __init__(self):
        self.IF = InstructionFetchState()  # Instruction Fetch state
        self.ID = InstructionDecodeState()  # Instruction Decode state
        self.EX = ExecutionState()  # Execution state
        self.MEM = MemoryAccessState()  # Memory Access state
        self.WB = WriteBackState()  # Write Back state

    def next(self):
        # Reset the state for the next instruction cycle
        self.ID = InstructionDecodeState()
        self.EX = ExecutionState()
        self.MEM = MemoryAccessState()
        self.WB = WriteBackState()


# Core implementation for five-stage pipeline
class Core_five(object):
    def __init__(self, ioDir, imem, dmem):
        self.myRF = RegisterFile(ioDir)  # Register file
        self.cycle = 0  # Cycle counter
        self.num_instr = 0  # Number of instructions executed
        self.halted = False  # Halt flag
        self.ioDir = ioDir  # Input-output directory
        self.state = State_five()  # Current state
        self.nextState = State_five()  # Next state
        self.ext_imem = imem  # Instruction memory
        self.ext_dmem = dmem  # Data memory


# Helper function to convert integer to binary string
def int2bin(x: int, n_bits: int = 32) -> str:
    bin_x = bin(x & (2**n_bits - 1))[2:]  # Binary representation
    return "0" * (n_bits - len(bin_x)) + bin_x


# Helper function to convert binary string to integer
def bin2int(x: str, sign_ext: bool = False) -> int:
    x = str(x)
    if sign_ext and x[0] == "1":  # Handle signed binary numbers
        return -(-int(x, 2) & (2 ** len(x) - 1))
    return int(x, 2)


# Instruction Fetch Stage
class InstructionFetchStage:
    def __init__(self, state: State_five, ins_mem: InsMem):
        self.state = state  # Current state
        self.ins_mem = ins_mem  # Instruction memory

    def run(self):
        # Perform instruction fetch if no hazards or halts
        if self.state.IF.nop or self.state.ID.nop or (self.state.ID.hazard_nop and self.state.EX.nop):
            return
        # Fetch instruction and reverse bits
        instr = self.ins_mem.read_instr(self.state.IF.PC)[::-1]
        if instr == "1" * 32:  # Check for HALT instruction
            self.state.IF.nop = True
            self.state.ID.nop = True
        else:
            self.state.ID.PC = self.state.IF.PC  # Pass PC to ID stage
            self.state.IF.PC += 4  # Increment PC by 4
            self.state.ID.instr = instr  # Pass instruction to ID stage

# -----------------------------------------
# Instruction Decode Stage


class InstructionDecodeStage:
    def __init__(self, state: State_five, rf: RegisterFile):
        self.state = state
        self.rf = rf

    def detect_hazard(self, rs):
        # Check for hazards in the pipeline
        if rs == self.state.MEM.write_reg_addr and not self.state.MEM.read_mem:
            # Forward from EX to Decode stage
            return 2
        elif rs == self.state.WB.write_reg_addr and self.state.WB.write_enable:
            # Forward from WB or MEM stage to Decode stage
            return 1
        elif rs == self.state.MEM.write_reg_addr and self.state.MEM.read_mem:
            # Stall the pipeline due to memory hazard
            self.state.ID.hazard_nop = True
            return 1
        else:
            return 0

    def read_data(self, rs, forward_signal):
        # Read data from the appropriate stage based on forwarding signal
        if forward_signal == 1:
            return self.state.WB.write_data
        elif forward_signal == 2:
            return self.state.MEM.alu_result
        else:
            return self.rf.read_RF(rs)

    def run(self):
        # Perform the instruction decode stage
        if self.state.ID.nop:
            if not self.state.IF.nop:
                self.state.ID.nop = False
            return

        self.state.EX.instr = self.state.ID.instr
        self.state.EX.is_I_type = False
        self.state.EX.read_mem = False
        self.state.EX.write_mem = False
        self.state.EX.write_enable = False
        self.state.ID.hazard_nop = False
        self.state.EX.write_reg_addr = "000000"

        opcode = self.state.ID.instr[:7][::-1]
        func3 = self.state.ID.instr[12:15][::-1]

        # Handle R-type instruction
        if opcode == "0110011":
            rs1 = self.state.ID.instr[15:20][::-1]
            rs2 = self.state.ID.instr[20:25][::-1]

            forward_signal_1 = self.detect_hazard(rs1)
            forward_signal_2 = self.detect_hazard(rs2)

            if self.state.ID.hazard_nop:
                self.state.EX.nop = True
                return

            self.state.EX.rs = rs1
            self.state.EX.rt = rs2
            self.state.EX.read_data_1 = self.read_data(rs1, forward_signal_1)
            self.state.EX.read_data_2 = self.read_data(rs2, forward_signal_2)
            self.state.EX.write_reg_addr = self.state.ID.instr[7:12][::-1]
            self.state.EX.write_enable = True

            func7 = self.state.ID.instr[25:][::-1]
            if func3 == "000":
                # Handle add or sub instruction
                self.state.EX.alu_op = "00"
                if func7 == "0100000":
                    self.state.EX.read_data_2 = int2bin(
                        -bin2int(self.state.EX.read_data_2, sign_ext=True)
                    )
            elif func3 == "111":
                # Handle and instruction
                self.state.EX.alu_op = "01"
            elif func3 == "110":
                # Handle or instruction
                self.state.EX.alu_op = "10"
            elif func3 == "100":
                # Handle xor instruction
                self.state.EX.alu_op = "11"

        # Handle I-type instructions
        elif opcode == "0010011" or opcode == "0000011":
            rs1 = self.state.ID.instr[15:20][::-1]

            forward_signal_1 = self.detect_hazard(rs1)

            if self.state.ID.hazard_nop:
                self.state.EX.nop = True
                return

            self.state.EX.rs = rs1
            self.state.EX.read_data_1 = self.read_data(rs1, forward_signal_1)
            self.state.EX.write_reg_addr = self.state.ID.instr[7:12][::-1]
            self.state.EX.is_I_type = True
            self.state.EX.imm = self.state.ID.instr[20:][::-1]
            self.state.EX.write_enable = True
            self.state.EX.read_mem = opcode == "0000011"

            if func3 == "000":
                self.state.EX.alu_op = "00"
            elif func3 == "111":
                self.state.EX.alu_op = "01"
            elif func3 == "110":
                self.state.EX.alu_op = "10"
            elif func3 == "100":
                self.state.EX.alu_op = "11"

        # Handle J-type (jump) instruction
        elif opcode == "1101111":
            self.state.EX.imm = (
                "0"
                + self.state.ID.instr[21:31]
                + self.state.ID.instr[20]
                + self.state.ID.instr[12:20]
                + self.state.ID.instr[31]
            )[::-1]
            self.state.EX.write_reg_addr = self.state.ID.instr[7:12][::-1]
            self.state.EX.read_data_1 = int2bin(self.state.ID.PC)
            self.state.EX.read_data_2 = int2bin(4)
            self.state.EX.write_enable = True
            self.state.EX.alu_op = "00"
            self.state.IF.PC = self.state.ID.PC + \
                bin2int(self.state.EX.imm, sign_ext=True)
            self.state.ID.nop = True

        # Handle B-type (branch) instruction
        elif opcode == "1100011":
            rs1 = self.state.ID.instr[15:20][::-1]
            rs2 = self.state.ID.instr[20:25][::-1]

            forward_signal_1 = self.detect_hazard(rs1)
            forward_signal_2 = self.detect_hazard(rs2)

            if self.state.ID.hazard_nop:
                self.state.EX.nop = True
                return

            self.state.EX.rs = rs1
            self.state.EX.rt = rs2
            self.state.EX.read_data_1 = self.read_data(rs1, forward_signal_1)
            self.state.EX.read_data_2 = self.read_data(rs2, forward_signal_2)
            diff = bin2int(self.state.EX.read_data_1, sign_ext=True) - bin2int(
                self.state.EX.read_data_2, sign_ext=True
            )

            self.state.EX.imm = (
                "0"
                + self.state.ID.instr[8:12]
                + self.state.ID.instr[25:31]
                + self.state.ID.instr[7]
                + self.state.ID.instr[31]
            )[::-1]

            if (diff == 0 and func3 == "000") or (diff != 0 and func3 == "001"):
                self.state.IF.PC = self.state.ID.PC + \
                    bin2int(self.state.EX.imm, sign_ext=True)
                self.state.ID.nop = True
                self.state.EX.nop = True
            else:
                self.state.EX.nop = True

        # Handle SW-type (store word) instruction
        elif opcode == "0100011":
            rs1 = self.state.ID.instr[15:20][::-1]
            rs2 = self.state.ID.instr[20:25][::-1]

            forward_signal_1 = self.detect_hazard(rs1)
            forward_signal_2 = self.detect_hazard(rs2)

            if self.state.ID.hazard_nop:
                self.state.EX.nop = True
                return

            self.state.EX.rs = rs1
            self.state.EX.rt = rs2
            self.state.EX.read_data_1 = self.read_data(rs1, forward_signal_1)
            self.state.EX.read_data_2 = self.read_data(rs2, forward_signal_2)

            self.state.EX.imm = (
                self.state.ID.instr[7:12] + self.state.ID.instr[25:]
            )[::-1]
            self.state.EX.is_I_type = True
            self.state.EX.write_mem = True
            self.state.EX.alu_op = "00"

        if self.state.IF.nop:
            self.state.ID.nop = True
        return 1


# Execution Stage
class ExecutionStage:
    def __init__(self, state: State_five):
        self.state = state

    def run(self):
        if self.state.EX.nop:
            if not self.state.ID.nop:
                self.state.EX.nop = False
            return

        operand_1 = self.state.EX.read_data_1
        operand_2 = (
            self.state.EX.read_data_2
            if not self.state.EX.is_I_type and not self.state.EX.write_mem
            else self.state.EX.imm
        )

        # Perform ALU operations
        if self.state.EX.alu_op == "00":  # ADD
            self.state.MEM.alu_result = int2bin(
                bin2int(operand_1, sign_ext=True) +
                bin2int(operand_2, sign_ext=True)
            )
        elif self.state.EX.alu_op == "01":  # AND
            self.state.MEM.alu_result = int2bin(
                bin2int(operand_1, sign_ext=True) & bin2int(
                    operand_2, sign_ext=True)
            )
        elif self.state.EX.alu_op == "10":  # OR
            self.state.MEM.alu_result = int2bin(
                bin2int(operand_1, sign_ext=True) | bin2int(
                    operand_2, sign_ext=True)
            )
        elif self.state.EX.alu_op == "11":  # XOR
            self.state.MEM.alu_result = int2bin(
                bin2int(operand_1, sign_ext=True) ^ bin2int(
                    operand_2, sign_ext=True)
            )

        # Pass data to the memory stage
        self.state.MEM.rs = self.state.EX.rs
        self.state.MEM.rt = self.state.EX.rt
        self.state.MEM.read_mem = self.state.EX.read_mem
        self.state.MEM.write_mem = self.state.EX.write_mem
        if self.state.EX.write_mem:
            self.state.MEM.store_data = self.state.EX.read_data_2
        self.state.MEM.write_enable = self.state.EX.write_enable
        self.state.MEM.write_reg_addr = self.state.EX.write_reg_addr

        if self.state.ID.nop:
            self.state.EX.nop = True


class MemoryAccessStage:
    def __init__(self, state: State_five, data_mem: DataMem):
        self.state = state  # State of the pipeline
        self.data_mem = data_mem  # Data memory interface

    def run(self):
        # Check if the memory access stage is in a no-operation (nop) state
        if self.state.MEM.nop:
            if not self.state.EX.nop:
                # Transition out of nop if EX stage is active
                self.state.MEM.nop = False
            return

        if self.state.MEM.read_mem:
            # Perform memory read operation and store result for write-back
            self.state.WB.write_data = self.data_mem.read_data_mem(
                self.state.MEM.alu_result
            )
        elif self.state.MEM.write_mem:
            # Perform memory write operation
            self.data_mem.write_data_mem(
                self.state.MEM.alu_result, self.state.MEM.store_data
            )
        else:
            # Pass ALU result directly to write-back if no memory operation
            self.state.WB.write_data = self.state.MEM.alu_result
            self.state.MEM.store_data = self.state.MEM.alu_result

        # Update write-back state with control signals and result
        self.state.WB.write_enable = self.state.MEM.write_enable
        self.state.WB.write_reg_addr = self.state.MEM.write_reg_addr

        # If EX stage is inactive, transition to nop state
        if self.state.EX.nop:
            self.state.MEM.nop = True


class WriteBackStage:
    def __init__(self, state: State_five, rf: RegisterFile):
        self.state = state  # State of the pipeline
        self.rf = rf  # Register file interface

    def run(self):
        # Check if the write-back stage is in a no-operation (nop) state
        if self.state.WB.nop:
            if not self.state.MEM.nop:
                # Transition out of nop if MEM stage is active
                self.state.WB.nop = False
            return

        if self.state.WB.write_enable:
            # Write data back to the register file
            self.rf.write_RF(self.state.WB.write_reg_addr,
                             self.state.WB.write_data)

        # If MEM stage is inactive, transition to nop state
        if self.state.MEM.nop:
            self.state.WB.nop = True

# -----------------------------------------
# Five-Stage Core Implementation


class FiveStageCore(Core_five):
    def __init__(self, ioDir, imem, dmem):
        super(FiveStageCore, self).__init__(ioDir + os.sep + "FS_", imem, dmem)
        self.opFilePath = ioDir + os.sep + "StateResult_FS.txt"

        # Initialize pipeline stages
        self.if_stage = InstructionFetchStage(self.state, self.ext_imem)
        self.id_stage = InstructionDecodeStage(self.state, self.myRF)
        self.ex_stage = ExecutionStage(self.state)
        self.mem_stage = MemoryAccessStage(self.state, self.ext_dmem)
        self.wb_stage = WriteBackStage(self.state, self.myRF)

    def step(self):
        # Perform a single cycle of the five-stage pipeline
        if (
            self.state.IF.nop
            and self.state.ID.nop
            and self.state.EX.nop
            and self.state.MEM.nop
            and self.state.WB.nop
        ):
            self.halted = True  # Check for pipeline halt condition

        current_instr = self.state.ID.instr  # Track current instruction in decode stage

        # Execute stages in reverse order to maintain data dependencies
        self.wb_stage.run()  # Write Back stage
        self.mem_stage.run()  # Memory Access stage
        self.ex_stage.run()  # Execution stage
        self.id_stage.run()  # Instruction Decode stage
        self.if_stage.run()  # Instruction Fetch stage

        # Output state of the register file after the cycle
        self.myRF.output_RF(self.cycle)
        # Print state of all pipeline stages
        self.printState(self.state, self.cycle)

        # Update instruction count and cycle count
        self.num_instr += int(current_instr != self.state.ID.instr)
        self.cycle += 1

    def printState(self, state, cycle):
        # Print the state of the pipeline after each cycle
        printstate = ["-" * 70 + "\n",
                      f"State after executing cycle: {cycle}\n"]

        # Append pipeline stage states to the print output
        printstate.append("\n")
        printstate.extend(
            [f"IF.{key}: {val}\n" for key, val in state.IF.__dict__().items()]
        )
        printstate.append("\n")
        printstate.extend(
            [f"ID.{key}: {val}\n" for key, val in state.ID.__dict__().items()]
        )
        printstate.append("\n")
        printstate.extend(
            [f"EX.{key}: {val}\n" for key, val in state.EX.__dict__().items()]
        )
        printstate.append("\n")
        printstate.extend(
            [f"MEM.{key}: {val}\n" for key,
                val in state.MEM.__dict__().items()]
        )
        printstate.append("\n")
        printstate.extend(
            [f"WB.{key}: {val}\n" for key, val in state.WB.__dict__().items()]
        )

        perm = "w" if cycle == 0 else "a"  # File writing mode
        with open(self.opFilePath, perm) as wf:
            wf.writelines(printstate)


# -----------------------------------------
# Performance Metrics Calculation Functions

# Single Stage Core Metrics
def single_metrics(opFilePath: str, ss: SingleStageCore):
    # Calculate performance metrics for single-stage core
    ss_metrics = [
        "Single Stage Core Performance Metrics: ",
        f"Number of Cycles taken:  {ss.cycle}",
        f"Cycles per instruction:  {int((ss.cycle - 1) / ss.inst)}",
        f"Instructions per cycle:  {int(ss.inst / (ss.cycle - 1))}",
    ]

    # Write metrics to output file
    with open(opFilePath + os.sep + "SingleMetrics.txt", "w") as f:
        f.write("\n".join(ss_metrics))


# Five Stage Core Metrics
def five_metrics(opFilePath: str, fs: FiveStageCore):
    # Calculate performance metrics for five-stage core
    fs_metrics = [
        "Five Stage Core Performance Metrics:",
        f"Number of Cycles taken:  {fs.cycle}",
        f"Cycles per instruction: {fs.cycle / fs.num_instr:.2f}",
        f"Instructions per cycle: {fs.num_instr / fs.cycle:.2f}",
    ]

    # Write metrics to output file
    with open(opFilePath + os.sep + "FiveMetrics.txt", "w") as f:
        f.write("\n".join(fs_metrics))


# Combined Performance Metrics
def Performance_metrics(opFilePath: str, ss: SingleStageCore, fs: FiveStageCore):
    # Compile metrics for both cores
    ss_metrics = [
        "Single Stage Core Performance Metrics: ",
        f"Number of Cycles taken:  {ss.cycle}",
        f"Cycles per instruction:  {int((ss.cycle - 1) / ss.inst)}",
        f"Instructions per cycle:  {int(ss.inst / (ss.cycle - 1))}",
    ]

    fs_metrics = [
        "Five Stage Core Performance Metrics:",
        f"Number of Cycles taken:  {fs.cycle}",
        f"Cycles per instruction: {fs.cycle / fs.num_instr:.2f}",
        f"Instructions per cycle: {fs.num_instr / fs.cycle:.2f}",
    ]

    # Write combined metrics to output file
    with open(opFilePath + os.sep + "PerformanceMetrics_Result.txt", "w") as f:
        f.write("\n".join(ss_metrics) + "\n\n" + "\n".join(fs_metrics))


# main
if __name__ == "__main__":

    # Parse command-line arguments to get input file directory
    parser = argparse.ArgumentParser(
        description='RV32I single and five-stage processor'
    )
    parser.add_argument('--iodir', default="",
                        type=str, help='Directory containing the input files.')
    args = parser.parse_args()

    # Get the absolute path for the input/output directory
    ioDir = os.path.abspath(args.iodir)

    # Display the directory being used for input files
    print("IO Directory:", ioDir)

    # Initialize shared instruction memory
    imem = InsMem("Imem", ioDir)

    # Initialize single-stage processor components
    dmem_ss = DataMem("SS", ioDir)
    ssCore = SingleStageCore(ioDir, imem, dmem_ss)

    # Run single-stage processor until halted
    while True:
        if not ssCore.halted:
            ssCore.step()
        if ssCore.halted:
            # Output register file after final cycle
            ssCore.myRF.outputRF(ssCore.cycle)
            # Output pipeline states after the final cycle
            ssCore.printState(ssCore.nextState, ssCore.cycle)
            ssCore.cycle += 1
            break

    # Save single-stage data memory to file
    dmem_ss.outputDataMem()

    # Initialize five-stage processor components
    dmem_fs = DataMem("FS", ioDir)
    fsCore = FiveStageCore(ioDir, imem, dmem_fs)

    # Run five-stage processor until halted
    while True:
        if not fsCore.halted:
            fsCore.step()
        if fsCore.halted:
            break

    # Save five-stage data memory to file
    dmem_fs.outputDataMem()

    # Print single-stage performance metrics to console
    print("Single Stage Core Performance Metrics: ")
    print("Number of Cycles taken: ", ssCore.cycle, end=", ")
    print("Number of Instructions in Imem: ", ssCore.inst, end="\n\n")

    # Print five-stage performance metrics to console
    print("Five Stage Core Performance Metrics: ")
    print("Number of Cycles taken: ", fsCore.cycle, end=", ")
    # Increment instruction count to account for the additional HALT instruction
    fsCore.num_instr += 1
    print("Number of Instructions in Imem: ", fsCore.num_instr, end="\n\n")

    # Save performance metrics to files
    Performance_metrics(ioDir, ssCore, fsCore)
    single_metrics(ioDir, ssCore)
    five_metrics(ioDir, fsCore)
