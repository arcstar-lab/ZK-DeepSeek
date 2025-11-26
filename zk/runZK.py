import asyncio
import numpy as np

zkDataDir = '../zkdata'

async def main():
    sem1 = asyncio.Semaphore(1)
    sem7 = asyncio.Semaphore(7)
    sem8 = asyncio.Semaphore(8)
    sem32 = asyncio.Semaphore(32)

    async def taskEmbed():
        print(f'taskEmbed')

        fEmbed = open("embed.log", "a", buffering=1)
        fEmbedErr = open("embedErr.log", "w", buffering=1)

        data = np.fromfile(f"{zkDataDir}/pos_0/tokens.bin", dtype=np.int64)
        print('xs: ', data)
        dataLen = len(data)

        # 计算 所有 vocabulary embedding 的 hash
        async def computeHash(tokenId):
            async with sem32:
                p = await asyncio.create_subprocess_exec("bash", "-lc", f'node build/src/index.js computeHash embed {tokenId}',
                                             stdout=fEmbed, stderr=fEmbedErr)
                rc = await p.wait()
                return (tokenId, rc)

        results = await asyncio.gather(*(computeHash(i) for i in range(0, 129280) ))

        # 汇集所有的 vocabulary embedding 到 hashTable.json 中
        p = await asyncio.create_subprocess_exec("bash", "-lc", f'node build/src/index.js precomputeHashes embed',
                                             stdout=fEmbed, stderr=fEmbedErr)
        rc = await p.wait()

        # 计算 tokens 的 root hash
        p = await asyncio.create_subprocess_exec("bash", "-lc", f'node build/src/index.js computeEmbedHash embed',
                                             stdout=fEmbed, stderr=fEmbedErr)
        rc = await p.wait()

        async def taskEmbedBase(rowId):
            async with sem7:
                p = await asyncio.create_subprocess_exec("bash", "-lc", f'node build/src/index.js embedSectionBase embed {rowId}',
                                             stdout=fEmbed, stderr=fEmbedErr)
                rc = await p.wait()
                return (rowId, rc)

        results = await asyncio.gather(*(taskEmbedBase(i) for i in range(0, dataLen) ))

        async def taskEmbedMerge(rowId):
            async with sem7:
                p = await asyncio.create_subprocess_exec("bash", "-lc", f'node build/src/index.js embedSectionMerge embed {rowId}',
                                             stdout=fEmbed, stderr=fEmbedErr)
                rc = await p.wait()
                return (rowId, rc)

        results = await asyncio.gather(*(taskEmbedMerge(i) for i in range(0, dataLen) ))

        async def taskEmbedRowsMerge():
            async with sem7:
                p = await asyncio.create_subprocess_exec("bash", "-lc", f'node build/src/index.js embedRowsMerge embed',
                                             stdout=fEmbed, stderr=fEmbedErr)
                rc = await p.wait()
                return rc

        results = await asyncio.gather((taskEmbedRowsMerge() ))

        fEmbed.close()
        fEmbedErr.close()

    async def taskAttnNorm(name, posId, layerId):
        print(f'taskAttnNorm {name}')

        fLog = open(f"{name}_Norm.log", "a", buffering=1)
        fErr = open(f"{name}_NormErr.log", "w", buffering=1)

        data = np.fromfile(f"{zkDataDir}/pos_0/tokens.bin", dtype=np.int64)
        print('xs: ', data)
        dataLen = len(data)

        async def taskAttnNormBase(rowId, ind):
            async with sem7:
                p = await asyncio.create_subprocess_exec("bash", "-lc", f'node build/src/index.js normBase {name} {posId} {layerId} {rowId} {ind}',
                                             stdout=fLog, stderr=fErr)
                rc = await p.wait()
                return (rowId, rc)

        if name == 'attn_norm':
            results = await asyncio.gather(*(taskAttnNormBase(i, j) for i in range(0, 24) for j in (0, 32)))
        elif name == 'q_norm':
            results = await asyncio.gather(*(taskAttnNormBase(i, 0) for i in range(0, 24)))

        async def taskAttnNormMerge(rowId, startIdx):
            async with sem8:
                rc = 0
                for j in range(startIdx, 0, -8):
                    p = await asyncio.create_subprocess_exec("bash", "-lc", f'node build/src/index.js normMerge {name} {posId} {layerId} {rowId} {j}',
                                                stdout=fLog, stderr=fErr)
                    rc = await p.wait()
                return (rowId, rc)

        if name == 'attn_norm':
            results = await asyncio.gather(*(taskAttnNormMerge(i, 62) for i in range(0, 24)))
        elif name == 'q_norm':
            results = await asyncio.gather(*(taskAttnNormMerge(i, 30) for i in range(0, 24)))

        async def normWrapRow():
            async with sem7:
                p = await asyncio.create_subprocess_exec("bash", "-lc", f'node build/src/index.js normWrapRow {name} {posId} {layerId}',
                                             stdout=fLog, stderr=fErr)
                rc = await p.wait()
                return rc

        results = await asyncio.gather(normWrapRow())

        async def normMergeRow():
            async with sem7:
                p = await asyncio.create_subprocess_exec("bash", "-lc", f'node build/src/index.js normMergeRow {name} {posId} {layerId}',
                                             stdout=fLog, stderr=fErr)
                rc = await p.wait()
                return rc

        results = await asyncio.gather(normMergeRow())

        fLog.close()
        fErr.close()


    # gate 中 experts 选择逻辑
    async def taskExpertSelector(name, posId, layerId):
        print(f'taskGateExpertSelector')

        fgate = open(f"{name}_expertSelector.log", "a", buffering=1)
        fgateErr = open(f"{name}_expertSelectorErr.log", "w", buffering=1)

        async def taskGroupBase(rowId):
            async with sem8:
                p = await asyncio.create_subprocess_exec("bash", "-lc", f'node build/src/index.js expertsGroupBase {name} {posId} {layerId} {rowId}',
                                             stdout=fgate, stderr=fgateErr)
                rc = await p.wait()
                return (rowId, rc)

        results = await asyncio.gather(*(taskGroupBase(i) for i in range(0, 24) ))

        async def taskGroupMerge(rowId):
            async with sem8:
                p = await asyncio.create_subprocess_exec("bash", "-lc", f'node build/src/index.js expertsGroupMerge {name} {posId} {layerId} {rowId}',
                                             stdout=fgate, stderr=fgateErr)
                rc = await p.wait()
                return (rowId, rc)

        results = await asyncio.gather(*(taskGroupMerge(i) for i in range(0, 24) ))

        async def taskSortedGroupBase(rowId):
            async with sem8:
                p = await asyncio.create_subprocess_exec("bash", "-lc", f'node build/src/index.js expertsSortedGroupBase {name} {posId} {layerId} {rowId}',
                                             stdout=fgate, stderr=fgateErr)
                rc = await p.wait()
                return (rowId, rc)

        results = await asyncio.gather(*(taskSortedGroupBase(i) for i in range(0, 24) ))

        async def taskSortedGroupMerge(rowId):
            async with sem8:
                p = await asyncio.create_subprocess_exec("bash", "-lc", f'node build/src/index.js expertsSortedGroupMerge {name}s {posId} {layerId} {rowId}',
                                             stdout=fgate, stderr=fgateErr)
                rc = await p.wait()
                return (rowId, rc)

        results = await asyncio.gather(*(taskSortedGroupMerge(i) for i in range(0, 24) ))

        async def taskSelectorBase(rowId):
            async with sem8:
                p = await asyncio.create_subprocess_exec("bash", "-lc", f'node build/src/index.js expertsSelectorBase {name} {posId} {layerId} {rowId}',
                                             stdout=fgate, stderr=fgateErr)
                rc = await p.wait()
                return (rowId, rc)

        results = await asyncio.gather(*(taskSelectorBase(i) for i in range(0, 24) ))

        async def taskSelectorMerge():
            async with sem8:
                p = await asyncio.create_subprocess_exec("bash", "-lc", f'node build/src/index.js expertsSelectorMerge {name} {posId} {layerId}',
                                             stdout=fgate, stderr=fgateErr)
                rc = await p.wait()
                return rc

        results = await asyncio.gather((taskSelectorMerge() ))

        fgate.close()
        fgateErr.close()


    async def taskRope_pe():
        print(f'taskRope')

        fLog = open("rope.log", "a", buffering=1)
        fErr = open("ropeErr.log", "w", buffering=1)

        data = np.fromfile(f"{zkDataDir}/pos_0/tokens.bin", dtype=np.int64)
        print('xs: ', data)
        dataLen = len(data)

        async def ropeBase(name, posId, layerId, rowId, ind, f_out, f_err):
            async with sem7:
                p = await asyncio.create_subprocess_exec("bash", "-lc", f'node build/src/index.js ropeBase {name} {posId} {layerId} {rowId} {ind}',
                                                        stdout=f_out, stderr=f_err)
                rc = await p.wait()
                return rc

        results = await asyncio.gather(*(ropeBase('q_pe', 0, 0, i, j, fLog, fErr) for i in range(0, 24) for j in (0, 32, 64, 96) ))

        async def ropeMerge(name, posId, layerId, rowId, ind, f_out, f_err):
            async with sem8:
                p = await asyncio.create_subprocess_exec("bash", "-lc", f'node  build/src/index.js ropeMerge {name} {posId} {layerId} {rowId} {ind}',
                                                        stdout=f_out, stderr=f_err)
                rc = await p.wait()
                return rc

        for j in range(126, -1, -8):
            results = await asyncio.gather(*(ropeMerge('q_pe', 0, 0, i, j, fLog, fErr) for i in range(0, 24) ))

        async def wrapRopeRow(name, posId, layerId, f_out, f_err):
            p = await asyncio.create_subprocess_exec("bash", "-lc", f'node  build/src/index.js wrapRopeRow {name} {posId} {layerId}',
                                                    stdout=f_out, stderr=f_err)
            rc = await p.wait()
            return rc

        results = await asyncio.gather(wrapRopeRow('q_pe', 0, 0, fLog, fErr))

        async def mergeRopeRow(name, posId, layerId, f_out, f_err):
            p = await asyncio.create_subprocess_exec("bash", "-lc", f'node  build/src/index.js mergeRopeRow {name} {posId} {layerId}',
                                                    stdout=f_out, stderr=f_err)
            rc = await p.wait()
            return rc

        results = await asyncio.gather(mergeRopeRow('q_pe', 0, 0, fLog, fErr))

        fLog.close()
        fErr.close()

    async def taskSoftmax(name):
        print(f'taskSoftmax {name}')

        fLog = open(f"{name}_softmax.log", "a", buffering=1)
        fErr = open(f"{name}_softmaxErr.log", "w", buffering=1)

        async def softmaxHeadBase(posId, layerId, rowId, headId, headDim):
            async with sem7:
                p = await asyncio.create_subprocess_exec(
                    "bash", "-lc",
                    f'node  build/src/index.js softmaxHeadBase {name} {posId} {layerId} {rowId} {headId} {headDim}',
                    stdout=fLog, stderr=fErr)
                rc = await p.wait()
                return rc
        results = await asyncio.gather(*(softmaxHeadBase(0, 0, i, j, 24) for i in range(0, 24) for j in range(0, 128, 4)))

        async def softmaxHeadMerge(posId, layerId, rowId, headDim):
            async with sem8:
                rc = 0
                for ind in range(126, -1, -8):
                    p = await asyncio.create_subprocess_exec(
                        "bash", "-lc",
                        f'node build/src/index.js softmaxHeadMerge {name} {posId} {layerId} {rowId} {ind} {headDim}',
                        stdout=fLog, stderr=fErr)
                    rc = await p.wait()
                return (rowId, rc)
        results = await asyncio.gather(*(softmaxHeadMerge(0, 0, i, 24) for i in range(0, 24)))

        async def softmaxWrapRow(posId, layerId):
            p = await asyncio.create_subprocess_exec("bash", "-lc", f'node  build/src/index.js softmaxWrapRow {name} {posId} {layerId}',
                                                    stdout=fLog, stderr=fErr)
            rc = await p.wait()
            return rc
        results = await asyncio.gather(softmaxWrapRow(0, 0))

        async def softmaxMergeRow(posId, layerId):
            p = await asyncio.create_subprocess_exec("bash", "-lc", f'node  build/src/index.js softmaxMergeRow {name} {posId} {layerId}',
                                                    stdout=fLog, stderr=fErr)
            rc = await p.wait()
            return rc
        results = await asyncio.gather(softmaxMergeRow(0, 0))

        fLog.close()
        fErr.close()

    async def taskSigmoid(name):
        print(f'taskSigmoid {name}')

        fLog = open(f"{name}_sigmoid.log", "a", buffering=1)
        fErr = open(f"{name}_sigmoidErr.log", "w", buffering=1)

        async def sigmoidSectionBase(posId, layerId, rowId):
            async with sem8:
                p = await asyncio.create_subprocess_exec("bash", "-lc", f'node build/src/index.js sigmoidSectionBase {name} {posId} {layerId} {rowId}',
                                                        stdout=fLog, stderr=fErr)
                rc = await p.wait()
                return rc

        results = await asyncio.gather(*(sigmoidSectionBase(0, 3, i) for i in range(0, 24) ))

        async def sigmoidSectionMerge(posId, layerId, rowId):
            async with sem8:
                p = await asyncio.create_subprocess_exec("bash", "-lc", f'node build/src/index.js sigmoidSectionMerge {name} {posId} {layerId} {rowId}',
                                             stdout=fLog, stderr=fErr)
                rc = await p.wait()
                return rc

        results = await asyncio.gather(*(sigmoidSectionMerge(0, 3, i) for i in range(0, 24) ))

        async def sigmoidRowBase(posId, layerId):
            p = await asyncio.create_subprocess_exec("bash", "-lc", f'node  build/src/index.js sigmoidRowBase {name} {posId} {layerId}',
                                                    stdout=fLog, stderr=fErr)
            rc = await p.wait()
            return rc
        results = await asyncio.gather(sigmoidRowBase(0, 3))

        async def sigmoidRowMerge(posId, layerId):
            p = await asyncio.create_subprocess_exec("bash", "-lc", f'node  build/src/index.js sigmoidRowMerge {name} {posId} {layerId}',
                                                    stdout=fLog, stderr=fErr)
            rc = await p.wait()
            return rc
        results = await asyncio.gather(sigmoidRowMerge(0, 3))

        fLog.close()
        fErr.close()

    async def taskGemm(name, posId, layerId, InDim, OutDim, ShortDim):
        print(f'taskGemm {name}')

        fLog = open(f"{name}_gemm.log", "a", buffering=1)
        fErr = open(f"{name}_gemmErr.log", "w", buffering=1)

        data = np.fromfile(f"{zkDataDir}/pos_{posId}/tokens.bin", dtype=np.int64)
        print('xs: ', data)
        rowCount = len(data)

        segmentCount = InDim // ShortDim
        startIndArr = [i * 32 for i in range(0, segmentCount // 32)]
        print('startIndArr: ', startIndArr)

        async def gemmXBase(rowId, ind):
            async with sem8:
                p = await asyncio.create_subprocess_exec("bash", "-lc", f'node  build/src/index.js gemmXBase {name} {posId} {layerId} {rowId} {ind}',
                                                        stdout=fLog, stderr=fErr)
                rc = await p.wait()
                return rc
        results = await asyncio.gather(*(gemmXBase(i, j) for i in range(0, rowCount) for j in startIndArr))

        async def gemmXMergeRow(ind):
            async with sem8:
                p = await asyncio.create_subprocess_exec("bash", "-lc", f'node build/src/index.js gemmXMergeRow {name} {posId} {layerId} {ind}',
                                                        stdout=fLog, stderr=fErr)
                rc = await p.wait()
                return rc
        results = await asyncio.gather(*(gemmXMergeRow(j) for j in range(segmentCount - 1, 2 * segmentCount - 1)))

        async def gemmWBase(rowId, ind):
            async with sem8:
                p = await asyncio.create_subprocess_exec("bash", "-lc", f'node build/src/index.js gemmWBase {name} {posId} {layerId} {rowId} {ind}',
                                                        stdout=fLog, stderr=fErr)
                rc = await p.wait()
                return rc
        results = await asyncio.gather(*(gemmWBase(i, j) for i in range(0, OutDim) for j in startIndArr ))

        async def gemmWMergeRow(ind):
            async with sem8:
                rc = 0
                for rowIndex in range(1, 512, 32):
                    p = await asyncio.create_subprocess_exec("bash", "-lc", f'node build/src/index.js gemmWMergeRow {name} {posId} {layerId} {ind} {rowIndex}',
                                                            stdout=fLog, stderr=fErr)
                    rc = await p.wait()
                return rc
        results = await asyncio.gather(*(gemmWMergeRow(j) for j in range(segmentCount - 1, 2 * segmentCount - 1) ))

        async def gemmXWBase(ind):
            async with sem8:
                p = await asyncio.create_subprocess_exec("bash", "-lc", f'node  build/src/index.js  gemmXWBase {name} {posId} {layerId} {ind}',
                                                        stdout=fLog, stderr=fErr)
                rc = await p.wait()
                return rc
        results = await asyncio.gather(*(gemmXWBase(i) for i in startIndArr))

        async def gemmXWMerge(ind):
            async with sem1:
                p = await asyncio.create_subprocess_exec("bash", "-lc", f'node  build/src/index.js  gemmXWMerge {name} {posId} {layerId} {ind}',
                                                        stdout=fLog, stderr=fErr)
                rc = await p.wait()
                return rc
        results = await asyncio.gather(*(gemmXWMerge(i) for i in range(segmentCount - 2, -1, -8)))

        fLog.close()
        fErr.close()

    # await taskExpertSelector_gate(0, 4)
    await taskEmbed()
    # await taskAttnNorm('attn_norm', 0, 0)
    # await taskAttnNorm('q_norm')
    # await taskRope_pe()
    # await taskSoftmax('scores')
    # await taskSigmoid('gate')
    # await taskExpertSelector('gate', 0, 3)
    # await taskGemm('wkv_a1', 0, 0, 7168, 512, 112)

    print("all done.")

asyncio.run(main())
