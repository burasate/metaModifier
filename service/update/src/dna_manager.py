# -*- coding: utf-8 -*-
import sys, os
import maya.cmds as cmds
import maya.OpenMaya as om

#https://epicgames.github.io/MetaHuman-DNA-Calibration/
class metahuman_dna_manager:
    def __init__(self):
        self.ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/dna_calibration'
        self.MAYA_VERSION = cmds.about(v=1)
        self.ROOT_LIB_DIR = f"{self.ROOT_DIR}/lib/Maya{self.MAYA_VERSION}"
        if not os.path.exists(self.ROOT_LIB_DIR):
            raise Warning('Maya {} doesn\'t support this tool\nplease use maya version that compatible with')
        if sys.platform == "win32":
            self.LIB_DIR = f"{self.ROOT_LIB_DIR}/windows"
        elif sys.platform == "linux":
            self.LIB_DIR = f"{self.ROOT_LIB_DIR}/linux"
        else:
            raise OSError(
                "OS not supported, please compile dependencies and add value to LIB_DIR"
            )
        for i in [self.ROOT_DIR, self.LIB_DIR]:
            if not i in sys.path:
                print(i)
                sys.path.insert(0, i)
        # ------------------------------------------------------------------------
        from dna import (
            BinaryStreamReader,
            BinaryStreamWriter,
            DataLayer_All,
            FileStream,
            Status,
        )
        from dnacalib import (
            CommandSequence,
            DNACalibDNAReader,
            SetNeutralJointRotationsCommand,
            SetNeutralJointTranslationsCommand,
            SetVertexPositionsCommand,
            VectorOperation_Add,
            PruneBlendShapeTargetsCommand,
            ClearBlendShapesCommand,
            SetBlendShapeTargetDeltasCommand,
        )
        from dna_viewer import DNA, RigConfig, build_rig, build_meshes
        # ------------------------------------------------------------------------
        self.BinaryStreamReader = BinaryStreamReader
        self.BinaryStreamWriter = BinaryStreamWriter
        self.DataLayer_All = DataLayer_All
        self.FileStream = FileStream
        self.Status = Status
        self.CommandSequence = CommandSequence
        self.DNACalibDNAReader = DNACalibDNAReader
        self.SetNeutralJointRotationsCommand = SetNeutralJointRotationsCommand
        self.SetNeutralJointTranslationsCommand = SetNeutralJointTranslationsCommand
        self.SetVertexPositionsCommand = SetVertexPositionsCommand
        self.VectorOperation_Add = VectorOperation_Add
        self.PruneBlendShapeTargetsCommand = PruneBlendShapeTargetsCommand
        self.ClearBlendShapesCommand = ClearBlendShapesCommand
        self.SetBlendShapeTargetDeltasCommand = SetBlendShapeTargetDeltasCommand
        self.DNA = DNA
        self.RigConfig = RigConfig
        self.build_rig = build_rig
        self.build_meshes = build_meshes
        # ------------------------------------------------------------------------
        self.OUTPUT_DIR = f"{self.ROOT_DIR}/output"
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        self.ROOT_LIB_DIR = f"{self.ROOT_DIR}/lib"
        self.DATA_DIR = f"{self.ROOT_DIR}/data"
        self.DNA_DIR = f"{self.DATA_DIR}/dna_files"
        self.ANALOG_GUI = f"{self.DATA_DIR}/analog_gui.ma"
        self.GUI = f"{self.DATA_DIR}/gui.ma"
        self.ADDITIONAL_ASSEMBLE_SCRIPT = f"{self.DATA_DIR}/additional_assemble_script.py"
        self.ADD_MESH_NAME_TO_BLEND_SHAPE_CHANNEL_NAME = True
        self.DNA_PATH, self.CHARACTER_NAME, self.MODIFIED_CHARACTER_DNA = [None]*3

    def dna_viewer_ui(self):
        import dna_viewer
        dna_viewer.show()

    def set_dna_path(self, path):
        self.DNA_PATH = path
        self.CHARACTER_NAME = os.path.basename(path).split('.')[0]
        self.MODIFIED_CHARACTER_DNA = self.OUTPUT_DIR + '/{}_modified'.format(self.CHARACTER_NAME)
        print(self.CHARACTER_NAME, self.DNA_PATH)

    def load_dna_reader(self, path):
        stream = self.FileStream(path, self.FileStream.AccessMode_Read, self.FileStream.OpenMode_Binary)
        reader = self.BinaryStreamReader(stream, self.DataLayer_All)
        reader.read()
        if not self.Status.isOk():
            status = self.Status.get()
            raise RuntimeError(f"Error loading DNA: {status.message}")
        return reader

    def save_dna(self, reader):
        if not self.MODIFIED_CHARACTER_DNA:
            raise Warning(self.MODIFIED_CHARACTER_DNA)
        stream = self.FileStream(
            f"{self.MODIFIED_CHARACTER_DNA}.dna",
            self.FileStream.AccessMode_Write,
            self.FileStream.OpenMode_Binary,
        )
        writer = self.BinaryStreamWriter(stream)
        writer.setFrom(reader)
        writer.write()

        if not self.Status.isOk():
            status = self.Status.get()
            raise RuntimeError(f"Error saving DNA: {status.message}")

    def get_mesh_vertex_positions_from_scene(self, meshName):
        try:
            sel = om.MSelectionList()
            sel.add(meshName)

            dag_path = om.MDagPath()
            sel.getDagPath(0, dag_path)

            mf_mesh = om.MFnMesh(dag_path)
            positions = om.MPointArray()

            mf_mesh.getPoints(positions, om.MSpace.kObject)
            return [
                [positions[i].x, positions[i].y, positions[i].z]
                for i in range(positions.length())
            ]
        except RuntimeError:
            print(f"{meshName} is missing, skipping it")
            return None

    def run_joints_command(self, reader, calibrated):
        joint_translations = []
        joint_rotations = []

        for i in range(reader.getJointCount()):
            joint_name = reader.getJointName(i)

            translation = cmds.xform(joint_name, query=True, translation=True)
            joint_translations.append(translation)

            rotation = cmds.joint(joint_name, query=True, orientation=True)
            joint_rotations.append(rotation)

        set_new_joints_translations = self.SetNeutralJointTranslationsCommand(joint_translations)
        set_new_joints_rotations = self.SetNeutralJointRotationsCommand(joint_rotations)

        commands = self.CommandSequence()
        commands.add(set_new_joints_translations)
        commands.add(set_new_joints_rotations)

        commands.run(calibrated)
        if not self.Status.isOk():
            status = self.Status.get()
            raise RuntimeError(f"Error run_joints_command: {status.message}")

    def run_vertices_command(
        self, calibrated, old_vertices_positions, new_vertices_positions, mesh_index
    ):
        deltas = []
        for new_vertex, old_vertex in zip(new_vertices_positions, old_vertices_positions):
            delta = []
            for new, old in zip(new_vertex, old_vertex):
                delta.append(new - old)
            deltas.append(delta)

        new_neutral_mesh = self.SetVertexPositionsCommand(
            mesh_index, deltas, self.VectorOperation_Add
        )
        commands = self.CommandSequence()
        commands.add(new_neutral_mesh)
        commands.run(calibrated)

        if not self.Status.isOk():
            status = self.Status.get()
            raise RuntimeError(f"Error run_vertices_command: {status.message}")

    def assemble_maya_scene(self):
        dna = self.DNA(f"{self.MODIFIED_CHARACTER_DNA}.dna")
        config = self.RigConfig(
            gui_path=f"{self.DATA_DIR}/gui.ma",
            analog_gui_path=f"{self.DATA_DIR}/analog_gui.ma",
            aas_path=self.ADDITIONAL_ASSEMBLE_SCRIPT,
            lod_filter=[0]
        )
        self.build_rig(dna=dna, config=config)

        cmds.file(rename=f"{self.MODIFIED_CHARACTER_DNA}.mb")
        cmds.file(save=True)


    def get_current_vertices_positions(self, dna):
        current_vertices_positions = {}
        for mesh_index, name in enumerate(dna.meshes.names):
            current_vertices_positions[name] = {
                "mesh_index": mesh_index,
                "positions": self.get_mesh_vertex_positions_from_scene(name),
            }
        return current_vertices_positions