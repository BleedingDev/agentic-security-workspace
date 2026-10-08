package nativeaot.objectmodel.net80;

import ghidra.app.util.bin.BinaryReader;
import ghidra.app.util.bin.MemoryByteProvider;
import static ghidra.app.util.bin.StructConverter.DWORD;
import static ghidra.app.util.bin.StructConverter.WORD;
import ghidra.program.model.address.Address;
import ghidra.program.model.data.*;
import ghidra.program.model.listing.GhidraClass;
import ghidra.program.model.symbol.SourceType;
import nativeaot.Constants;
import nativeaot.objectmodel.ElementType;
import nativeaot.objectmodel.MethodTable;

public class MethodTableNet80 extends MethodTable {
    private static final int ELEMENT_TYPE_MASK = 0x7C000000;
    private static final int ELEMENT_TYPE_SHIFT = 26;

    private int _flags;
    private int _baseSize;
    private long _relatedTypeAddress;
    private int _hashCode;
    private long[] _vtable;
    private long[] _interfaceSlots;

    public MethodTableNet80(MethodTableManagerNet80 manager, Address address) {
        super(manager, address);
    }

    public int getFlags() {
        return _flags;
    }

    @Override
    public int getElementType() {
        return (_flags & ELEMENT_TYPE_MASK) >>> ELEMENT_TYPE_SHIFT;
    }

    @Override
    public int getBaseSize() {
        return _baseSize;
    }

    @Override
    public int getDataSize() {
        return _baseSize 
            - 0x8 /* obj header */
            - 0x8 /* mt */;
    }

    @Override
    public long getRelatedTypeAddress() {
        return _relatedTypeAddress;
    }

    public int getHashCode() {
        return _hashCode;
    }

    @Override
    public long[] getVTable() {
        return _vtable;
    }

    @Override
    public int getVTableSlotCount() {
        return _vtable.length;
    }

    @Override
    public long getVTableSlot(int index) {
        return _vtable[index];
    }

    @Override
    public long[] getInterfaceAddresses() {
        return _interfaceSlots;
    }
    
    @Override
    protected Structure constructMTType() throws Exception {
        var result = new StructureDataType(getMTName(), 0);
        result.setCategoryPath(Constants.CATEGORY_METHOD_TABLES);
        result.setDescription(getAddress().toString());

        // https://github.com/dotnet/runtime/blob/a3b2d40328be0be3f8dd43f0e5cc925b8827b044/src/coreclr/nativeaot/Runtime/inc/MethodTable.h#L104-L115
        result.add(DWORD, "uFlags", null);
        result.add(DWORD, "uBaseSize", null);
        result.add(Pointer64DataType.dataType, "relatedType", null);
        result.add(WORD, "usNumVtableSlots", null);
        result.add(WORD, "usNumInterfaces", null);
        result.add(DWORD, "uHashCode", null);

        int count = 0;
        for (var chunk : getVTableChunks()) {
            result.add(chunk.getOrCreateDataType(), String.format(VTABLE_MEMBER_FORMAT, count++), null);
        }

        if (_interfaceSlots.length > 0) {
            result.add(new ArrayDataType(Pointer64DataType.dataType, _interfaceSlots.length), "Interfaces", null);
        }

        return result;
    }

    @Override
    public void initFromMemory() throws Exception {
        var address = getAddress();

        var reader = new BinaryReader(
            new MemoryByteProvider(
                getManager().getProgram().getMemory(),
                address.getAddressSpace()
            ),
            true
        );
        reader.setPointerIndex(address.getOffset());

        _flags = reader.readNextInt();
        _baseSize = reader.readNextInt();
        _relatedTypeAddress = reader.readNextLong();
        int vtableSlotCount = reader.readNextShort();
        int interfacesCount = reader.readNextShort();
        _hashCode = reader.readNextInt();

        if (vtableSlotCount < 0 || vtableSlotCount >= 1000)
            throw new IllegalArgumentException("Invalid VTable slot count");
        if (interfacesCount < 0 || interfacesCount >= 1000)
            throw new IllegalArgumentException("Invalid interface count");

        _vtable = new long[vtableSlotCount];
        for (int i = 0; i < vtableSlotCount; i++) {
            _vtable[i] = reader.readNextLong();
        }

        _interfaceSlots = new long[interfacesCount];
        for (int i = 0; i < interfacesCount; i++) {
            _interfaceSlots[i] = reader.readNextLong();
        }

        // Do some validation to see if the MT makes sense.
        var elementType = getElementType();
        if (elementType == ElementType.INTERFACE) {
            if (_baseSize != 0x00) {
                throw new Exception("Unexpected non-zero interface base size.");
            } else if (getRelatedTypeAddress() != 0) {
                throw new Exception("Unexpected non-zero interface related type.");
            }
        } else if (_baseSize < 0x10) { // TODO: abstract away
            throw new Exception("Unexpected base size.");
        }
    }
}