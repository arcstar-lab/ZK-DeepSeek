import { Field, ZkProgram, verify, Provable, Struct, Poseidon, Int64, UInt64, UInt32, Bool, SelfProof, Cache, VerificationKey } from 'o1js';
import { promises as fs } from "node:fs";
import { Command } from "commander";

function nowPrefix() {
  const d = new Date();
  const pad = (n: number) => n.toString().padStart(2, "0");

  const year = d.getFullYear();
  const month = pad(d.getMonth() + 1);
  const day = pad(d.getDate());
  const hour = pad(d.getHours());
  const minute = pad(d.getMinutes());
  const second = pad(d.getSeconds());

  return `${year}-${month}-${day} ${hour}:${minute}:${second}`;
}

const EmbedsInOneFile = 256;
const BytesOfAnEmbed = 8 * 7168;
// const BytesOfAnEmbed = 4 * 7168;
const BytesOfAFile = EmbedsInOneFile * BytesOfAnEmbed;

const zkDataDir = "../zkdata"

const EmbedsDir = `${zkDataDir}/embeds/`;

// Begin Embed ============================================

function createEmbedClass(name: string, Dim: number, ShortDim: number) {
  const ShortDimCount = Dim / ShortDim;

  const VocabSize = 129280;

    const RowSectionX = Provable.Array(Int64, ShortDim);

    class RowSectionInput extends Struct({
        rowId: UInt32,

        left: UInt32,
        right: UInt32,
        zStart: Field,
        zEnd: Field,

        z: Field,
    }) {}

    class RowSectionOutput extends Struct({
      hash: Field,
      zsum: Field,
    }) {}

    const RowSection = ZkProgram({
      name: 'RowSection',
      publicInput: RowSectionInput,
      publicOutput: RowSectionOutput,
      methods: {
        base: {
          privateInputs: [RowSectionX],
          async method(input: RowSectionInput, xs: Int64[]) {
              let h = Poseidon.hashPacked(RowSectionX, xs);

              let zsum = Field(0);
              let zi: Field = input.zStart;
              for(let i = 0; i < ShortDim; i++) {
                zsum = zsum.add(zi.mul(xs[i].toField()));
                zi = zi.mul(input.z);
              }
              zi.assertEquals(input.zEnd);

              const out = new RowSectionOutput({
                hash: h,
                zsum: zsum,
              });
              return {publicOutput: out};
          },
        },
        merge: {
          privateInputs: [SelfProof, SelfProof],
          async method(input: RowSectionInput,
            leftProof: InstanceType<typeof SelfProof<RowSectionInput,RowSectionOutput> > ,
            rightProof: InstanceType<typeof SelfProof<RowSectionInput,RowSectionOutput> >) {
              leftProof.verify();
              rightProof.verify();

              let leftInput = leftProof.publicInput;
              let leftOutput = leftProof.publicOutput;
              let rightInput = rightProof.publicInput;
              let rightOutput = rightProof.publicOutput;

              input.rowId.assertEquals(leftInput.rowId);
              input.rowId.assertEquals(rightInput.rowId);

              input.left.assertEquals(leftInput.left);
              input.right.assertEquals(rightInput.right);
              leftInput.right.assertEquals(rightInput.left);

              input.zStart.assertEquals(leftInput.zStart, 'zStart not equal');
              input.zEnd.assertEquals(rightInput.zEnd, 'zEnd not equal');
              leftInput.zEnd.assertEquals(rightInput.zStart, 'zStart zEnd not equal');

              input.z.assertEquals(leftInput.z, 'z 1 not equal');
              input.z.assertEquals(rightInput.z, 'z 2 not equal');

              let h = Poseidon.hash([leftOutput.hash, rightOutput.hash]);
              let zsum = leftOutput.zsum.add(rightOutput.zsum);

              const out = new RowSectionOutput({
                hash: h,
                zsum: zsum,
              });
              return {publicOutput: out};
          },
        },
      },
    });
    const RowSectionProof = ZkProgram.Proof(RowSection);
    type RowSectionProofType = InstanceType<typeof RowSectionProof>;
    type RowSectionProofJSON = ReturnType<InstanceType<typeof RowSectionProof>["toJSON"]>;

    async function compileEmbedSection() {
        const cache = Cache.FileSystem(`./o1js-cache/Embed-section`);

        return RowSection.compile({cache: cache});
    }

    async function sectionBase(name: string, tokenList: number[], rowId: number, vkEmbed: VerificationKey) {
        let xs: Int64[] = await getAnEmbedFromFile(BigInt(tokenList[rowId]));

        let zStr: string = await readFile('proofs/embed/hash');
        let z = Field(zStr);

        for(let i = 0; i < ShortDimCount; i++) {
            let left = i * ShortDim;
            let right = (i + 1) * ShortDim;
            let zStart = fastPow(z, Dim * rowId + left);
            let zEnd = fastPow(z, Dim * rowId + right);

            let input = new RowSectionInput({
                rowId: UInt32.from(rowId),
                left: UInt32.from(left),
                right: UInt32.from(right),
                zStart,
                zEnd,
                z,
            })

            let xArr = xs.slice(left, right);

            // console.log(`${nowPrefix()} xArr.length: ${xArr.length}, wArr.length: ${wArr.length}, qArr.length: ${qArr.length}, rArr.length: ${rArr.length}`)
            const proof = await RowSection.base(input, xArr);
            // const proof = await NormX.base(input, xArr, wArr, qArr);

            const start = performance.now();
            const ok = await verify(proof.proof, vkEmbed);
            const end = performance.now();

            let proofStr = JSON.stringify(proof.proof.toJSON());
            fs.mkdir(`proofs/embed/row_${rowId}`, { recursive: true }).then(() =>
                fs.writeFile(`proofs/embed/row_${rowId}/base_${ShortDimCount-1 + i}.json`, proofStr, "utf8")
            ).catch(console.error);

            console.log(`${nowPrefix()} Embed Section row_${rowId} base proof ${ShortDimCount-1 + i} verify result: ${ok}, verifying time: ${end - start} ms`);
        }
    }

    async function sectionMerge(name: string, rowId: number, vkEmbed: VerificationKey) {
        let proofs: RowSectionProofType[] = new Array(2 * ShortDimCount - 1);
        for (let i = 0; i < ShortDimCount; i++) {
            let proofFile = `proofs/embed/row_${rowId}/base_${ShortDimCount-1 + i}.json`;
            let proofStr = await readFile(proofFile);
            const proofJson = JSON.parse(proofStr) as RowSectionProofJSON;
            const proof = await RowSectionProof.fromJSON(proofJson);
            // const ok = await verify(proof, vkNorm); 
            // assert(ok, `base proof base_${rowId}_${j} verify failed!`);
            proofs[ShortDimCount - 1 + i] = proof;
            // console.log(`${nowPrefix()} add base_${ShortDimCount-1 + j}.json to proofs ${ShortDimCount - 1 + j}`);
        }

        for(let j = ShortDimCount-2; j >= 0; j--) {
            let leftProof = proofs[2 * j + 1];
            let rightProof = proofs[2 * j + 2];
            let input = new RowSectionInput({
                rowId: UInt32.from(rowId),
                left: leftProof.publicInput.left,
                right: rightProof.publicInput.right,
                zStart: leftProof.publicInput.zStart,
                zEnd: rightProof.publicInput.zEnd,
                z: leftProof.publicInput.z,
            })

            const proof = await RowSection.merge(input, leftProof, rightProof);

            const start = performance.now();
            const ok = await verify(proof.proof, vkEmbed);
            const end = performance.now();

            console.log(`${nowPrefix()} Embed RowSection row_${rowId} merge proof ${j} verify result: ${ok}, verifying time: ${end - start} ms`);
            proofs[j] = proof.proof;

            let proofStr = JSON.stringify(proof.proof.toJSON())
            await fs.writeFile(`proofs/embed/row_${rowId}/merge_${j}.json`, proofStr, "utf8");
        }
    }

    class RowsInput extends Struct({
        rowStart: UInt32,
        rowEnd: UInt32,
        zStart: Field,
        zEnd: Field,
        z: Field,
    }) {}

    const Rows = ZkProgram({
      name: 'Rows',
      publicInput: RowsInput,
      publicOutput: RowSectionOutput,
      methods: {
        base: {
          privateInputs: [RowSectionProof, RowSectionProof],
          async method(input: RowsInput, leftProof: RowSectionProofType, rightProof: RowSectionProofType) {
              leftProof.verify();
              rightProof.verify();

              let h = Poseidon.hash([leftProof.publicOutput.hash, rightProof.publicOutput.hash]);
              let zsum = leftProof.publicOutput.zsum.add(rightProof.publicOutput.zsum);

              let leftInput = leftProof.publicInput;
              let rightInput = rightProof.publicInput;

              input.rowStart.assertEquals(leftInput.rowId);
              input.rowEnd.assertEquals(rightInput.rowId.add(1));

              leftInput.left.assertEquals(UInt32.from(0));
              leftInput.right.assertEquals(UInt32.from(Dim));
              rightInput.left.assertEquals(UInt32.from(0));
              rightInput.right.assertEquals(UInt32.from(Dim));

              input.zStart.assertEquals(leftInput.zStart, 'zStart not equal');
              leftInput.zEnd.assertEquals(rightInput.zStart, 'zStart zEnd not equal');
              input.zEnd.assertEquals(rightInput.zEnd, 'zEnd not equal');

              input.z.assertEquals(leftInput.z);
              input.z.assertEquals(rightInput.z);

              const out = new RowSectionOutput({
                hash: h,
                zsum: zsum,
              });
              return {publicOutput: out};
          },
        },

        single: {
          privateInputs: [RowSectionProof],
          async method(input: RowsInput, leftProof: RowSectionProofType) {
              leftProof.verify();

              let h = leftProof.publicOutput.hash;
              let zsum = leftProof.publicOutput.zsum;

              let leftInput = leftProof.publicInput;

              input.rowStart.assertEquals(leftInput.rowId);
              input.rowEnd.assertEquals(leftInput.rowId.add(1));

              leftInput.left.assertEquals(UInt32.from(0));
              leftInput.right.assertEquals(UInt32.from(Dim));

              input.zStart.assertEquals(leftInput.zStart);
              input.zEnd.assertEquals(leftInput.zEnd);

              input.z.assertEquals(leftInput.z);

              const out = new RowSectionOutput({
                hash: h,
                zsum: zsum,
              });
              return {publicOutput: out};
          },
        },

        merge: {
          privateInputs: [SelfProof, SelfProof],
          async method(input: RowsInput,
            leftProof: InstanceType<typeof SelfProof<RowsInput,RowSectionOutput> > ,
            rightProof: InstanceType<typeof SelfProof<RowsInput,RowSectionOutput> >) {
              leftProof.verify();
              rightProof.verify();

              let leftInput = leftProof.publicInput;
              let leftOutput = leftProof.publicOutput;
              let rightInput = rightProof.publicInput;
              let rightOutput = rightProof.publicOutput;

              input.rowStart.assertEquals(leftInput.rowStart);
              leftInput.rowEnd.assertEquals(rightInput.rowStart)
              input.rowEnd.assertEquals(rightInput.rowEnd);

              input.zStart.assertEquals(leftInput.zStart, 'zStart not equal');
              input.zEnd.assertEquals(rightInput.zEnd, 'zEnd not equal');
              leftInput.zEnd.assertEquals(rightInput.zStart, 'zStart zEnd not equal');

              input.z.assertEquals(leftInput.z, 'z 1 not equal');
              input.z.assertEquals(rightInput.z, 'z 2 not equal');

              let h = Poseidon.hash([leftOutput.hash, rightOutput.hash]);
              let zsum = leftOutput.zsum.add(rightOutput.zsum);

              const out = new RowSectionOutput({
                hash: h,
                zsum: zsum,
              });
              return {publicOutput: out};
          },
        },
      },
    });
    // const { verificationKey: vkRows } = await Rows.compile({cache: cache});
    const RowsProof = ZkProgram.Proof(Rows);
    type RowsProofType = InstanceType<typeof RowsProof>;
    type RowsProofJSON = ReturnType<InstanceType<typeof RowsProof>["toJSON"]>;

    async function compileEmbedRows() {
        const cache = Cache.FileSystem(`./o1js-cache/Embed-rows`);

        await compileEmbedSection();

        return Rows.compile({cache: cache});
    }

    async function rowsMerge(name: string, tokenListLen: number, vkEmbed: VerificationKey) {
        let proofs: RowSectionProofType[] = [];
        for (let i = 0; i < tokenListLen; i++) {
            let proofFile = `proofs/embed/row_${i}/merge_0.json`;
            let proofStr = await readFile(proofFile);
            const proofJson = JSON.parse(proofStr) as RowSectionProofJSON;
            const proof = await RowSectionProof.fromJSON(proofJson);
            proofs.push(proof);
        }

        let rowsProofs: RowsProofType[] = [];
        for (let i = 0; i < Math.floor(tokenListLen / 2); i++) {
            let upProof = proofs[2 * i];
            let downProof = proofs[2 * i + 1];
            let rowsInput = new RowsInput({
                rowStart: upProof.publicInput.rowId,
                rowEnd: downProof.publicInput.rowId.add(1),
                zStart: upProof.publicInput.zStart,
                zEnd: downProof.publicInput.zEnd,
                z: upProof.publicInput.z,
            })

            const proof = await Rows.base(rowsInput, upProof, downProof);

            const start = performance.now();
            const ok = await verify(proof.proof, vkEmbed);
            const end = performance.now();

            console.log(`${nowPrefix()} Rows base proof ${i} verify result: ${ok}, verifying time: ${end - start} ms`);

            rowsProofs.push(proof.proof);
        }

        if(tokenListLen % 2 == 1) {
            let upProof = proofs[tokenListLen - 1];
            let rowsInput = new RowsInput({
              rowStart: upProof.publicInput.rowId,
              rowEnd: upProof.publicInput.rowId.add(1),
              zStart: upProof.publicInput.zStart,
              zEnd: upProof.publicInput.zEnd,
              z: upProof.publicInput.z,
            })

            const proof = await Rows.single(rowsInput, upProof);

            const start = performance.now();
            const ok = await verify(proof.proof, vkEmbed);
            const end = performance.now();

            console.log(`${nowPrefix()} Rows single proof ${rowsProofs.length - 1} verify result: ${ok}, verifying time: ${end - start} ms`);

            rowsProofs.push(proof.proof);
        }

        while(rowsProofs.length > 1) {
            let rowsProofs2: RowsProofType[]  = new Array(Math.floor((rowsProofs.length + 1) / 2));
            if(rowsProofs.length % 2 == 1) {
              rowsProofs2[rowsProofs2.length - 1] = rowsProofs[rowsProofs.length - 1];
              console.log(`${nowPrefix()} add nodes ${rowsProofs2.length - 1} directly`);
            }

            for(let i = 0; i < Math.floor(rowsProofs.length / 2); i++) {
                let upProof = rowsProofs[2 * i];
                let downProof = rowsProofs[2 * i + 1];
                let rowsInput = new RowsInput({
                  rowStart: upProof.publicInput.rowStart,
                  rowEnd: downProof.publicInput.rowEnd,
                  zStart: upProof.publicInput.zStart,
                  zEnd: downProof.publicInput.zEnd,
                  z: upProof.publicInput.z,
                })
                const proof = await Rows.merge(rowsInput, upProof, downProof);

                const start = performance.now();
                const ok = await verify(proof.proof, vkEmbed);
                const end = performance.now();

                console.log(`${nowPrefix()} Rows merge proof ${i} verify result: ${ok}, verifying time: ${end - start} ms`);
                rowsProofs2[i] = proof.proof;
            }
            rowsProofs = rowsProofs2;
            // console.log('nodes length: ' + nodes.length);
        }

        let resProof = rowsProofs[0];
        let proofStr =  JSON.stringify(resProof.toJSON())
        await fs.writeFile(`proofs/embed/embedInputs.json`, proofStr, "utf8");
    }

    async function computeHash(tokenId: bigint) {
        let hashDatas: Field[] = new Array(2 * ShortDimCount - 1);

        let embed = await getAnEmbedFromFile(tokenId);
        for(let i = ShortDimCount - 1; i < 2 * ShortDimCount - 1; i++) {
            let rowIndex = i + 1 - ShortDimCount;
            let data = embed.slice(rowIndex * ShortDim, (rowIndex + 1) * ShortDim);
            hashDatas[i] = Poseidon.hashPacked(RowSectionX, data);
        }

        for(let i = ShortDimCount - 2; i >= 0; i--) {
            let left = 2 * i + 1;
            let right = 2 * i + 2;
            hashDatas[i] = Poseidon.hash([hashDatas[left], hashDatas[right]]);
        }
        // return hashDatas[0];

        let hres = fieldToHex(hashDatas[0]);
        console.log(`${nowPrefix()} The hash of token ${tokenId}: ${hashDatas[0]}`);
        fs.mkdir(`proofs/embed/hashes`, { recursive: true }).then(() =>
          fs.writeFile(`proofs/embed/hashes/${tokenId}.json`, hres, "utf8")
        ).catch(console.error);
    }

    async function precomputeHashes() {
        let hashes: string[] = new Array(VocabSize);
        for(let i = 0; i < VocabSize; i++) {
            let hStr = await readFile(`proofs/embed/hashes/${i}.json`);
            hashes[i] = hStr;
        }

        await fs.writeFile(`proofs/embed/hashTable.json`, JSON.stringify(hashes, null, 2), "utf8");
    }

    function computeRootHash(hashes: string[]) {
      let nodes: Field[] = [];
      for(let i = 0; i < hashes.length; i++) {
        let v = Field(hashes[i]);
        nodes.push(v);
        console.log(`${nowPrefix()} nodes ${i}: ${v}`);
      }

      while(nodes.length > 1) {
        let nodes2: Field[] = new Array(Math.floor((nodes.length + 1) / 2));
        if(nodes.length % 2 == 1) {
          nodes2[nodes2.length - 1] = nodes[nodes.length - 1];
          console.log(`${nowPrefix()} nodes ${nodes2.length - 1}: ${nodes[nodes.length - 1]}`);
        }

        for(let i = 0; i < Math.floor(nodes.length / 2); i++) {
          let left = nodes[2 * i];
          let right = nodes[2 * i + 1];
          // console.log(`left node ${2 * i}: ${left}`);
          // console.log(`right node ${2 * i + 1}: ${right}`);
          nodes2[i] = Poseidon.hash([left, right]);
          console.log(`${nowPrefix()} nodes ${i}: ${nodes2[i]}`);
        }

        nodes = nodes2;
        // console.log('nodes length: ' + nodes.length);
      }
      return nodes[0];
    }

    async function computeEmbedHash(tokenIds: number[]) {
        try {
            const hashes = await readJsonFile<string[]>('proofs/embed/hashTable.json');
            console.log(Field(hashes[0]));

            let tokenHashes = new Array(tokenIds.length);
            for(let i = 0; i < tokenIds.length; i++) {
              let tokenId = tokenIds[i];
              tokenHashes[i] = hashes[tokenId];
            }
        
            let resHash = computeRootHash(tokenHashes);
            let hexHash = fieldToHex(resHash);
            console.log(`${nowPrefix()} resHash: ${hexHash} ${resHash}`);
            await fs.writeFile(`proofs/embed/hash`, hexHash, "utf8");
          } catch (err) {
            console.error('Error:', err);
          }
    }

    return {RowSection, RowSectionProof, compileEmbedSection, sectionBase, sectionMerge,
      Rows, RowsProof, compileEmbedRows, rowsMerge, computeHash, precomputeHashes, computeEmbedHash}
}

// End Embed ============================================

// Begin Norm ============================================

const TWO31 = Field(1n << 31n);
const TWO32 = Field(1n << 32n);

function fastPow(base: Field, exponent: number): Field {
  let result = Field(1);
  let currentBase = base;

  while (exponent > 0) {
    if (exponent % 2 === 1) {
      result = result.mul(currentBase);  // 如果是奇数，乘上当前基数
    }
    currentBase = currentBase.mul(currentBase);  // 基数平方
    exponent = Math.floor(exponent / 2); // 将指数除以 2
  }

  return result;
}

function asSigned32(u: UInt32): Field {
  const f = u.value;                          // [0, 2^32)
  const isNeg: Bool = f.greaterThanOrEqual(TWO31);
  return Provable.if(isNeg, f.sub(TWO32), f); // ≥2^31 看作负数：f - 2^32
}

const BigInt_TWO31 = BigInt(1) << 31n;
const BigInt_TWO32 = BigInt(1) << 32n;

function asInt64(u: UInt32): Int64 {
  const f = u.toBigint();                          // [0, 2^32)
  if(f >= BigInt_TWO31) {
    return Int64.from(f - BigInt_TWO32);
  } else {
    return Int64.from(f)
  }
}

function createNormClass(name: string, rescale: bigint, Dim: number, ShortDim: number) {
  const ShortDimCount = Dim / ShortDim;

  const RowSectionInt64 = Provable.Array(Int64, ShortDim);
  const RowSectionUInt64 = Provable.Array(UInt64, ShortDim);

  class NormXInput extends Struct({
    rowId: UInt32,

    left: UInt32,
    right: UInt32,

    zStart: Field,
    zEnd: Field,

    z: Field,

    rms: UInt64,
  }) {}

  class NormXOutput extends Struct({
    squareSum: Field,
    wHash: Field,
    zsumX: Field,
    zsumY: Field,
  }) {}

  class NormRowsInput extends Struct({
    rowStart: UInt32,
    rowEnd: UInt32,

    zStart: Field,
    zEnd: Field,

    z: Field,
  }) {}

  class NormRowsOutput extends Struct({
    wHash: Field,
    zsumX: Field,
    zsumY: Field,
  }) {}

  const NormX = ZkProgram({
    name: 'NormX',
    publicInput: NormXInput,
    publicOutput: NormXOutput,
    methods: {
      base: {
        // x, w, q(商), r(余数)
        privateInputs: [RowSectionInt64, RowSectionInt64, RowSectionInt64, RowSectionUInt64],
        // privateInputs: [FieldArr, FieldArr, FieldArr, FieldArr],
        // async method(input: NormXInput, xs: Int64[], ws: UInt32[], qs: Int64[], rs: Int64[]) {
        async method(input: NormXInput, xs: Int64[], ws: Int64[], qs: Int64[], rs: UInt64[]) {
          input.right.assertEquals(input.left.add(ShortDim));

          let wHash = Poseidon.hashPacked(RowSectionInt64, ws);
          // let wHash = Poseidon.hash(ws);
          let squareSum = Field(0);
          let zi = input.zStart;
          let zsumX = Field(0);
          let zsumY = Field(0);
          for(let i = 0; i < ShortDim; i++) {
            let x = xs[i];
            let w = ws[i];
            let q = qs[i];
            let r = rs[i];
            // let minusR = rs[i].neg();

            let rms = input.rms;
            squareSum = squareSum.add(x.mul(x).toField());
            zsumX = zsumX.add(zi.mul(x.toField()));
            zsumY = zsumY.add(zi.mul(q.toField()));
            zi = zi.mul(input.z);
            x.mul(w).assertEquals(q.mul(rms).add(r), `${i}: x * w != rms * q + r`);
            // r.mul(r).assertLessThan(rms.mul(rms), '|r| !< rms');
            r.assertLessThan(rms, 'r !< rms');
            // Provable
            // Bool.or(r.lessThan(rms), minusR.lessThan(rms)).assertTrue();
            // minusR.lessThan(rms);
          }
          zi.assertEquals(input.zEnd, 'rowStartZEnd not match with rowStartZStart');

          const out = new NormXOutput({
            squareSum: squareSum,
            wHash: wHash,
            zsumX: zsumX,
            zsumY: zsumY,
          });
          return {publicOutput: out};
        },
      },
      merge: {
        privateInputs: [SelfProof, SelfProof],
        async method(input: NormXInput,
          leftProof: InstanceType<typeof SelfProof<NormXInput, NormXOutput> > ,
          rightProof: InstanceType<typeof SelfProof<NormXInput, NormXOutput> >) {
            leftProof.verify();
            rightProof.verify();

            let leftInput = leftProof.publicInput;
            let leftOutput = leftProof.publicOutput;
            let rightInput = rightProof.publicInput;
            let rightOutput = rightProof.publicOutput;

            input.rowId.assertEquals(leftInput.rowId);
            input.rowId.assertEquals(rightInput.rowId);

            input.left.assertEquals(leftInput.left);
            input.right.assertEquals(rightInput.right);
            leftInput.right.assertEquals(rightInput.left);

            input.zStart.assertEquals(leftInput.zStart, 'zStart not equal');
            input.zEnd.assertEquals(rightInput.zEnd, 'zEnd not equal');
            leftInput.zEnd.assertEquals(rightInput.zStart, 'zStart zEnd not equal');

            input.z.assertEquals(leftInput.z, 'z 1 not equal');
            input.z.assertEquals(rightInput.z, 'z 2 not equal');

            input.rms.assertEquals(leftInput.rms);
            input.rms.assertEquals(rightInput.rms);

            let squareSum = leftOutput.squareSum.add(rightOutput.squareSum);
            let wHash = Poseidon.hash([leftOutput.wHash, rightOutput.wHash]);
            let zsumX = leftOutput.zsumX.add(rightOutput.zsumX);
            let zsumY = leftOutput.zsumY.add(rightOutput.zsumY);

            const out = new NormXOutput({
              squareSum: squareSum,
              wHash: wHash,
              zsumX: zsumX,
              zsumY: zsumY,
            });
            return {publicOutput: out};
        },
      },
    },
  });

  const NormXWithRescale = ZkProgram({
    name: 'NormX',
    publicInput: NormXInput,
    publicOutput: NormXOutput,
    methods: {
      base: {
        // x, w, q(商), r(余数)
        privateInputs: [RowSectionInt64, RowSectionInt64, RowSectionUInt64, RowSectionInt64, RowSectionUInt64],
        // privateInputs: [FieldArr, FieldArr, FieldArr, FieldArr, FieldArr],
        async method(input: NormXInput, xs: Int64[], ws: Int64[], prevRs: UInt64[], qs: Int64[], rs: UInt64[]) {
        // async method(input: NormXInput, xs: Field[], ws: Field[], prevRs: Field[], qs: Field[], rs: Field[]) {
          input.right.assertEquals(input.left.add(ShortDim));

          let wHash = Poseidon.hashPacked(RowSectionInt64, ws);
          // let wHash = Poseidon.hash(ws);
          let squareSum = Field(0);
          let zi = input.zStart;
          let zsumX = Field(0);
          let zsumY = Field(0);
          for(let i = 0; i < ShortDim; i++) {
            let x = xs[i];
            let w = ws[i];
            let q = qs[i];
            let r = rs[i];
            let prevR = prevRs[i];

            let prevY = x.mul(rescale).add(prevR);

            let rms = input.rms;
            squareSum = squareSum.add(x.mul(x).toField());
            zsumX = zsumX.add(zi.mul(prevY.toField()));
            zsumY = zsumY.add(zi.mul(q.toField()));
            zi = zi.mul(input.z);
            x.mul(w).assertEquals(q.mul(rms).add(r), `${i}: x * w != rms * q + r`);
            r.mul(r).assertLessThan(rms.mul(rms), '|r| !< rms');
          }
          zi.assertEquals(input.zEnd, 'rowStartZEnd not match with rowStartZStart');

          const out = new NormXOutput({
            squareSum: squareSum,
            wHash: wHash,
            zsumX: zsumX,
            zsumY: zsumY,
          });
          return {publicOutput: out};
        },
      },
      merge: {
        privateInputs: [SelfProof, SelfProof],
        async method(input: NormXInput,
          leftProof: InstanceType<typeof SelfProof<NormXInput, NormXOutput> > ,
          rightProof: InstanceType<typeof SelfProof<NormXInput, NormXOutput> >) {
            leftProof.verify();
            rightProof.verify();
  
            let leftInput = leftProof.publicInput;
            let leftOutput = leftProof.publicOutput;
            let rightInput = rightProof.publicInput;
            let rightOutput = rightProof.publicOutput;

            input.rowId.assertEquals(leftInput.rowId);
            input.rowId.assertEquals(rightInput.rowId);

            input.left.assertEquals(leftInput.left);
            input.right.assertEquals(rightInput.right);
            leftInput.right.assertEquals(rightInput.left);

            input.zStart.assertEquals(leftInput.zStart, 'zStart not equal');
            input.zEnd.assertEquals(rightInput.zEnd, 'zEnd not equal');
            leftInput.zEnd.assertEquals(rightInput.zStart, 'zStart zEnd not equal');

            input.z.assertEquals(leftInput.z, 'z 1 not equal');
            input.z.assertEquals(rightInput.z, 'z 2 not equal');

            input.rms.assertEquals(leftInput.rms);
            input.rms.assertEquals(rightInput.rms);

            let squareSum = leftOutput.squareSum.add(rightOutput.squareSum);
            let wHash = Poseidon.hash([leftOutput.wHash, rightOutput.wHash]);
            let zsumX = leftOutput.zsumX.add(rightOutput.zsumX);
            let zsumY = leftOutput.zsumY.add(rightOutput.zsumY);

            const out = new NormXOutput({
              squareSum: squareSum,
              wHash: wHash,
              zsumX: zsumX,
              zsumY: zsumY,
            });
            return {publicOutput: out};
        },
      },
    },
  });

  const NormXProof = rescale == 0n ? ZkProgram.Proof(NormX) : ZkProgram.Proof(NormXWithRescale);
  type NormXProofType = InstanceType<typeof NormXProof>;
  type NormXProofJSON = ReturnType<InstanceType<typeof NormXProof>["toJSON"]>;

  async function compileNormXWithCache() {
    const cache = Cache.FileSystem(`./o1js-cache/Norm-${name}`);
    if(rescale == 0n) {
      return NormX.compile({cache: cache});
    } else {
      return NormXWithRescale.compile({cache: cache});
    }
  }

  async function calcBase(name: string, posId: number, layerId: number, rowId: number, ind: number, vkNorm: VerificationKey) {
    let xs: Int64[][] = [];
    const bufX = await readBinary(`${zkDataDir}/pos_${posId}/layer_${layerId}/${name}_x.bin`);
    const xData = bufferToInt64ArrayLE(bufX);
    for(let i = 0; i < xData.length; i += Dim) {
      xs.push(xData.slice(i, i + Dim));
    }

    const bufW = await readBinary(`${zkDataDir}/pos_${posId}/layer_${layerId}/${name}_weight.bin`);
    let ws = bufferToUInt32ArrayLE(bufW);

    const bufRms = await readBinary(`${zkDataDir}/pos_${posId}/layer_${layerId}/${name}_rms.bin`);
    let rms0 = bufferToInt64ArrayLE(bufRms);

    let lenXs = xs.length;
    rms0 = rms0.slice(0, lenXs);

    let rms = rms0.map(x => UInt64.from(x.toBigint()));

    let zStr: string = await readFile('proofs/embed/hash');
    let z = Field(zStr);

    let wsInt64: Int64[] = new Array(Dim);
    let qs: Int64[] = new Array(Dim);
    let rs: Int64[] = new Array(Dim);

    for(let j = 0; j < Dim; j++) {
      let x = xs[rowId][j].toBigint();

      wsInt64[j] = asInt64(ws[j]);
      let w = wsInt64[j].toBigint();

      let prod = x * w;
      let rmsValue = rms[rowId].toBigInt()

      // 整型除法（对有符号整数）：向零截断（truncate toward 0）; 取模 %（余数）：与被除数（左操作数）同号。
      qs[j] = Int64.from(prod / rmsValue);
      rs[j] = Int64.from(prod % rmsValue);
      if(prod % rmsValue < 0) {
        qs[j] = qs[j].sub(1);
        rs[j] = rs[j].add(rmsValue);
      }
    }

    let prevRs: Int64[][] = [];
    if(rescale != 0n) {
      const bufPrevR = await readBinary(`${zkDataDir}/pos_${posId}/layer_${layerId}/${name}_r.bin`);
      const rData = bufferToInt64ArrayLE(bufPrevR);
      for(let i = 0; i < rData.length; i += Dim) {
        prevRs.push(rData.slice(i, i + Dim));
      }
    }

    for(let j = ind; j < ind + 32; j++) {
      let left = j * ShortDim;
      let right = (j + 1) * ShortDim;

      let zStart = fastPow(z, Dim * rowId + left);
      let zEnd = fastPow(z, Dim * rowId + right);

      let input = new NormXInput({
        rowId: UInt32.from(rowId),

        left: UInt32.from(left),
        right: UInt32.from(right),

        zStart: zStart,
        zEnd: zEnd,

        z: z,

        rms: rms[rowId],
      })

      let xArr = xs[rowId].slice(left, right);
      let wArr = wsInt64.slice(left, right);
      let qArr = qs.slice(left, right);
      let rArr = rs.slice(left, right).map(x => UInt64.from(x.toBigint()));
      // console.log(`xArr.length: ${xArr.length}, wArr.length: ${wArr.length}, qArr.length: ${qArr.length}, rArr.length: ${rArr.length}`)

      let proofStr = '';
      if(rescale == 0n) {
        const proof = await NormX.base(input, xArr, wArr, qArr, rArr);

        const start = performance.now();
        const ok = await verify(proof.proof, vkNorm);
        const end = performance.now();

        console.log(`${nowPrefix()} ${name} NormX row_${rowId} base proof ${ShortDimCount-1 + j} verify result: ${ok}, rescale: ${rescale}, verifying time: ${end - start} ms`);
        proofStr = JSON.stringify(proof.proof.toJSON());
      } else {
        let prevRArr = prevRs[rowId].slice(left, right).map(x => UInt64.from(x.toBigint()));
        const proof = await NormXWithRescale.base(input, xArr, wArr, prevRArr, qArr, rArr);
        const start = performance.now();
        const ok = await verify(proof.proof, vkNorm);
        const end = performance.now();
        console.log(`${nowPrefix()} ${name} NormX row_${rowId} base proof ${ShortDimCount-1 + j} verify result: ${ok}, rescale: ${rescale}, verifying time: ${end - start} ms`);
        proofStr = JSON.stringify(proof.proof.toJSON());
      }
      
      fs.mkdir(`proofs/pos_${posId}/layer_${layerId}/${name}/row_${rowId}`, { recursive: true }).then(() =>
          fs.writeFile(`proofs/pos_${posId}/layer_${layerId}/${name}/row_${rowId}/base_${ShortDimCount-1 + j}.json`, proofStr, "utf8")
        ).catch(console.error);
    }
  }

  async function calcMerge(name: string, posId: number, layerId: number, rowId: number, ind: number, vkNorm: VerificationKey) {
    let NormXProof = ZkProgram.Proof(NormX);
    if(rescale != 0n) {
      NormXProof = ZkProgram.Proof(NormXWithRescale);
    }

    let proofs: NormXProofType[] = new Array(2 * ShortDimCount - 1);
    for (let j = 0; j < ShortDimCount; j++) {
        let proofFile = `proofs/pos_${posId}/layer_${layerId}/${name}/row_${rowId}/base_${ShortDimCount-1 + j}.json`;
        let proofStr = await readFile(proofFile);
        const proofJson = JSON.parse(proofStr) as NormXProofJSON;
        const proof = await NormXProof.fromJSON(proofJson);
        // const ok = await verify(proof, vkNorm); 
        // assert(ok, `base proof base_${rowId}_${j} verify failed!`);
        proofs[ShortDimCount - 1 + j] = proof;
        // console.log(`add base_${ShortDimCount-1 + j}.json to proofs ${ShortDimCount - 1 + j}`);
    }

    for(let j = ind+1; j < ShortDimCount - 1; j++) {
      let proofFile = `proofs/pos_${posId}/layer_${layerId}/${name}/row_${rowId}/merge_${j}.json`;
      let proofStr = await readFile(proofFile);
      const proofJson = JSON.parse(proofStr) as NormXProofJSON;
      const proof = await NormXProof.fromJSON(proofJson);
      // const ok = await verify(proof, vkNorm); 
      // assert(ok, `merge proof merge_${rowId}_${j} verify failed!`);
      proofs[j] = proof;
      // console.log(`add merge_${j}.json to proofs ${j}`);
    }

    let count = 8;
    for(let j = ind; j >= 0; j--) {
      if(count == 0) break;
      count--;

      let leftProof = proofs[2 * j + 1];
      let rightProof = proofs[2 * j + 2];
      let normInput = new NormXInput({
        rowId: leftProof.publicInput.rowId,
        left: leftProof.publicInput.left,
        right: rightProof.publicInput.right,
        zStart: leftProof.publicInput.zStart,
        zEnd: rightProof.publicInput.zEnd,
        z: leftProof.publicInput.z,
        rms: leftProof.publicInput.rms
      })

      // console.log(leftProof.publicInput.right);
      // console.log(rightProof.publicInput.left);

      let proofStr = '';
      if(rescale == 0n) {
        const proof = await NormX.merge(normInput, leftProof, rightProof);

        const start = performance.now();
        const ok = await verify(proof.proof, vkNorm);
        const end = performance.now();

        console.log(`${nowPrefix()} ${name} NormX row_${rowId} merge proof ${j} verify result: ${ok}, rescale: ${rescale}, verifying time: ${end - start} ms`);
        proofs[j] = proof.proof;
        proofStr = JSON.stringify(proof.proof.toJSON())
      } else {
        const proof = await NormXWithRescale.merge(normInput, leftProof, rightProof);

        const start = performance.now();
        const ok = await verify(proof.proof, vkNorm);
        const end = performance.now();

        console.log(`${nowPrefix()} ${name} NormX row_${rowId} merge proof ${j} verify result: ${ok}, rescale: ${rescale}, verifying time: ${end - start} ms`);
        proofs[j] = proof.proof;
        proofStr = JSON.stringify(proof.proof.toJSON())
      }

      await fs.writeFile(`proofs/pos_${posId}/layer_${layerId}/${name}/row_${rowId}/merge_${j}.json`, proofStr, "utf8");
    }
  }

  const NormRows = ZkProgram({
    name: 'NormRows',
    publicInput: NormRowsInput,
    publicOutput: NormRowsOutput,
    methods: {
      base: {
        privateInputs: [NormXProof, Field, Field],
        // squareSumDivDimQ = (Σ xi^2) / Dim
        // squareSumDivDimR = (Σ xi^2) % Dim
        async method(input: NormRowsInput, proof: NormXProofType, squareSumDivDimQ: Field, squareSumDivDimR: Field) {
          proof.verify();

          // 检查 rms 计算公式
          let rms = proof.publicInput.rms;
          let rmsSquare = squareSumDivDimQ.add(1);
          rms.mul(rms).value.assertLessThanOrEqual(rmsSquare, 'rms^2 should <= rmsSquare');
          let rmsPlus1 = rms.add(1);
          rmsPlus1.mul(rmsPlus1).value.assertGreaterThan(rmsSquare, '(rms+1)^2 should > rmsSquare');
          proof.publicOutput.squareSum.assertEquals(squareSumDivDimQ.mul(Dim).add(squareSumDivDimR), 'q * Dim + r != squareSum');
          squareSumDivDimR.mul(squareSumDivDimR).assertLessThan(Dim * Dim, 'r should < Dim');
          // squareSumDivDimR.neg().assertLessThan(Dim, '-r should < Dim');

          proof.publicInput.left.assertEquals(UInt32.from(0));
          proof.publicInput.right.assertEquals(UInt32.from(Dim));

          input.rowStart.assertEquals(proof.publicInput.rowId);
          input.rowEnd.assertEquals(input.rowStart.add(1));

          let zn = fastPow(input.z, Dim);
          input.zStart.assertEquals(proof.publicInput.zStart, 'zStart not equal');
          input.zEnd.assertEquals(input.zStart.mul(zn), 'zEnd not equal');

          input.z.assertEquals(proof.publicInput.z, 'z not equal');

          const out = new NormRowsOutput({
            wHash: proof.publicOutput.wHash,
            zsumX: proof.publicOutput.zsumX,
            zsumY: proof.publicOutput.zsumY,
          });
          return {publicOutput: out};
        }
      },
      merge: {
        privateInputs: [SelfProof, SelfProof],
        async method(input: NormRowsInput,
          upProof: InstanceType<typeof SelfProof<NormRowsInput, NormRowsOutput> > ,
          downProof: InstanceType<typeof SelfProof<NormRowsInput, NormRowsOutput> >) {
            upProof.verify();
            downProof.verify();

            let upInput = upProof.publicInput;
            let upOutput = upProof.publicOutput;
            let downInput = downProof.publicInput;
            let downOutput = downProof.publicOutput;

            input.rowStart.assertEquals(upInput.rowStart);
            input.rowEnd.assertEquals(downInput.rowEnd);
            upInput.rowEnd.assertEquals(downInput.rowStart)

            input.zStart.assertEquals(upInput.zStart, 'zStart not equal');
            input.zEnd.assertEquals(downInput.zEnd, 'zEnd not equal');
            upInput.zEnd.assertEquals(downInput.zStart, 'zStart zEnd not equal');
  
            input.z.assertEquals(upInput.z, 'z 1 not equal');
            input.z.assertEquals(downInput.z, 'z 2 not equal');

            upOutput.wHash.assertEquals(downOutput.wHash, 'wHash not equal');

            let zsumX = upOutput.zsumX.add(downOutput.zsumX);
            let zsumY = upOutput.zsumY.add(downOutput.zsumY);

            const out = new NormRowsOutput({
              wHash: upOutput.wHash,
              zsumX: zsumX,
              zsumY: zsumY,
            });
            return {publicOutput: out};
        },
      },
    }
  });

  const NormRowsProof = ZkProgram.Proof(NormRows);
  type NormRowsProofType = InstanceType<typeof NormRowsProof>;
  type NormRowsProofJSON = ReturnType<InstanceType<typeof NormRowsProof>["toJSON"]>;

  async function compileNormRowsWithCache() {
    const cache = Cache.FileSystem(`./o1js-cache/NormRows-${name}`);

    await compileNormXWithCache();
    return NormRows.compile({cache: cache});
  }

  async function wrapRow(name: string, tokenListLen: number, posId: number, layerId: number, vkNormRows: VerificationKey) {
    for(let rowId = 0; rowId < tokenListLen; rowId++) {
      let proofFile = `proofs/pos_${posId}/layer_${layerId}/${name}/row_${rowId}/merge_0.json`;
      let proofStr = await readFile(proofFile);
      const proofJson = JSON.parse(proofStr) as NormXProofJSON;
      const earlierProof = await NormXProof.fromJSON(proofJson);
      // const ok = await verify(proof, vkNorm); 
      // assert(ok, `merge proof merge_${rowId}_${j} verify failed!`);

      let publicInput = earlierProof.publicInput;
      let rId = publicInput.rowId;

      let input = new NormRowsInput({
        rowStart: rId,
        rowEnd: rId.add(1),
        zStart: publicInput.zStart,
        zEnd: publicInput.zEnd,
        z: publicInput.z })

      let squareSum = earlierProof.publicOutput.squareSum;
      let q = squareSum.toBigInt() / BigInt(Dim);
      let r = squareSum.toBigInt() % BigInt(Dim);

      const proof = await NormRows.base(input, earlierProof, q, r);

      const start = performance.now();
      const ok = await verify(proof.proof, vkNormRows);
      const end = performance.now();

      console.log(`${nowPrefix()} NormRow base proof verify: ${ok}, rescale: ${rescale}, verifying time: ${end - start} ms`);

      let proofResStr = JSON.stringify(proof.proof.toJSON());
      fs.mkdir(`proofs/pos_${posId}/layer_${layerId}/${name}/summary`, { recursive: true }).then(() =>
        fs.writeFile(`proofs/pos_${posId}/layer_${layerId}/${name}/summary/wrap_row_${rowId}.json`, proofResStr, "utf8")
      ).catch(console.error);

      console.log(`${nowPrefix()} ${name} NormRows row_${rowId} wrap proof verify result: ${ok}, rescale: ${rescale}`);
    }
  }

  async function mergeRow(name: string, tokenListLen: number, posId: number, layerId: number, vkNormRows: VerificationKey) {
    let rowsProofs: NormRowsProofType[] = new Array(tokenListLen);
    for (let i = 0; i < tokenListLen; i++) {
        let proofFile = `proofs/pos_${posId}/layer_${layerId}/${name}/summary/wrap_row_${i}.json`;
        let proofStr = await readFile(proofFile);
        const proofJson = JSON.parse(proofStr) as NormRowsProofJSON;
        const proof = await NormRowsProof.fromJSON(proofJson);
        // const ok = await verify(proof, vkNorm); 
        // assert(ok, `base proof base_${rowId}_${j} verify failed!`);
        rowsProofs[i] = proof;
        // console.log(`add base_${ShortDimCount-1 + j}.json to proofs ${ShortDimCount - 1 + j}`);
    }

    while(rowsProofs.length > 1) {
      let rowsProofs2: NormRowsProofType[]  = new Array(Math.floor((rowsProofs.length + 1) / 2));
      if(rowsProofs.length % 2 == 1) {
        rowsProofs2[rowsProofs2.length - 1] = rowsProofs[rowsProofs.length - 1];
        console.log(`${nowPrefix()} add proofs2 ${rowsProofs2.length - 1} directly`);
      }

      for(let i = 0; i < Math.floor(rowsProofs.length / 2); i++) {
        let left = rowsProofs[2 * i];
        let right = rowsProofs[2 * i + 1];
        let rowsInput = new NormRowsInput({
          rowStart: left.publicInput.rowStart,
          rowEnd: right.publicInput.rowEnd,
          zStart: left.publicInput.zStart,
          zEnd: right.publicInput.zEnd,
          z: left.publicInput.z,
        })

        const proof = await NormRows.merge(rowsInput, left, right);

        const start = performance.now();
        const ok = await verify(proof.proof, vkNormRows);
        const end = performance.now();

        console.log(`${nowPrefix()} ${name} NormRows merge proof ${i} verify result: ${ok}, rescale: ${rescale}, verifying time: ${end - start} ms`);
        rowsProofs2[i] = proof.proof;
      }

      rowsProofs = rowsProofs2;
      // console.log('nodes length: ' + nodes.length);
    }

    const proof = rowsProofs[0];
    const ok = await verify(proof, vkNormRows);
    console.log(`${nowPrefix()} ${name} NormRows merge proof verify result: ${ok}`);

    let proofStr = JSON.stringify(proof.toJSON())
    fs.mkdir(`proofs/pos_${posId}/layer_${layerId}/${name}/summary`, { recursive: true }).then(() =>
      fs.writeFile(`proofs/pos_${posId}/layer_${layerId}/${name}/summary/${name}.json`, proofStr, "utf8")
    ).catch(console.error);
  }


  return { NormX, NormXProof, compileNormXWithCache, NormRows, NormRowsProof, compileNormRowsWithCache, calcBase, calcMerge, wrapRow, mergeRow };
}




// Begin Gemm ===========================================================================================

function createGemmClass(name: string, InDim: number, OutDim: number, ShortDim: number) {
  const ShortDimCount = InDim / ShortDim;

  const FieldArr = Provable.Array(Field, ShortDim);

  const ZSumArr = Provable.Array(Field, ShortDim);

  class GemmXInput extends Struct({
    rowStart: UInt32,
    rowEnd: UInt32,
    zkRowStart: Field,
    zkRowEnd: Field,
    znRowStart: Field,
    znRowEnd: Field,

    left: UInt32,
    right: UInt32,
    zLeft: Field,
    zRight: Field,

    z: Field,
    // zk: Field,
    // zn: Field,
  }) {}

  class GemmXOutput extends Struct({
    zsumX: Field,
    ZXs: ZSumArr,
  }) {}

  const GemmX = ZkProgram({
    name: 'GemmX',
    publicInput: GemmXInput,
    publicOutput: GemmXOutput,
    methods: {
      base: {
        privateInputs: [FieldArr,],
        async method(input: GemmXInput, xs: Field[]) {
          input.right.assertEquals(input.left.add(ShortDim));

          let zsumX = Field(0);
          let ZXs: Field[] = new Array(ShortDim);

          let zi = input.zkRowStart.mul(input.zLeft);
          let z = input.z;
          for(let i = 0; i < ShortDim; i++) {
            ZXs[i] = input.znRowStart.mul(xs[i]);  // Z^n * x[i]
            zsumX = zsumX.add(zi.mul(xs[i]));  // Z^(k + i) * x[i]
            zi = zi.mul(z);
          }
          zi.assertEquals(input.zkRowStart.mul(input.zRight), 'zLeft not match with zRight');

          const zk = fastPow(z, InDim);
          // zk.assertEquals(input.zk, 'zk not equal');
          input.zkRowEnd.assertEquals(input.zkRowStart.mul(zk), 'zkRowStart zkRowEnd not match');

          const zn = fastPow(z, OutDim);
          // zn.assertEquals(input.zn, 'zn not equal');
          input.znRowEnd.assertEquals(input.znRowStart.mul(zn), 'znRowStart znRowEnd not match');

          const out = new GemmXOutput({
            zsumX: zsumX,
            ZXs: ZXs,
          });
          return {publicOutput: out};
        },
      },
      merge: {
        privateInputs: [SelfProof, SelfProof],
        async method(input: GemmXInput,
          upProof: InstanceType<typeof SelfProof<GemmXInput, GemmXOutput> > ,
          downProof: InstanceType<typeof SelfProof<GemmXInput, GemmXOutput> >) {
            upProof.verify();
            downProof.verify();

            let upInput = upProof.publicInput;
            let upOutput = upProof.publicOutput;
            let downInput = downProof.publicInput;
            let downOutput = downProof.publicOutput;

            input.rowStart.assertEquals(upInput.rowStart);
            input.rowEnd.assertEquals(downInput.rowEnd);
            upInput.rowEnd.assertEquals(downInput.rowStart);

            input.zkRowStart.assertEquals(upInput.zkRowStart, 'zkRowStart not equal');
            input.zkRowEnd.assertEquals(downInput.zkRowEnd, 'zkRowEnd not equal');
            upInput.zkRowEnd.assertEquals(downInput.zkRowStart, 'zkRowStart zkRowEnd not equal');

            input.znRowStart.assertEquals(upInput.znRowStart, 'znRowStart not equal');
            input.znRowEnd.assertEquals(downInput.znRowEnd, 'znRowEnd not equal');
            upInput.znRowEnd.assertEquals(downInput.znRowStart, 'znRowStart znRowEnd not equal');

            input.left.assertEquals(upInput.left);
            input.left.assertEquals(downInput.left);
            input.right.assertEquals(upInput.right);
            input.right.assertEquals(downInput.right);

            input.zLeft.assertEquals(upInput.zLeft, 'zLeft 1 not equal');
            input.zLeft.assertEquals(downInput.zLeft, 'zLeft 2 not equal');
            input.zRight.assertEquals(upInput.zRight, 'zRight 1 not equal');
            input.zRight.assertEquals(downInput.zRight, 'zRight 2 not equal');

            input.z.assertEquals(upInput.z, 'z 1 not equal');
            input.z.assertEquals(downInput.z, 'z 2 not equal');

            // input.zn.assertEquals(upInput.zn, 'zn 1 not equal');
            // input.zn.assertEquals(downInput.zn, 'zn 2 not equal');

            let zsumX = upOutput.zsumX.add(downOutput.zsumX);
            let ZXs: Field[] = new Array(ShortDim);
            for(let i = 0; i < ShortDim; i++) {
              ZXs[i] = upOutput.ZXs[i].add(downOutput.ZXs[i]);
            }

            const out = new GemmXOutput({
              zsumX: zsumX,
              ZXs: ZXs,
            });
            return {publicOutput: out};
        },
      },
    },
  });
  // const { verificationKey: vkGemmX } = await GemmX.compile({cache: cache});
  // console.log(await GemmX.analyzeMethods());
  const GemmXProof = ZkProgram.Proof(GemmX);
  type GemmXProofType = InstanceType<typeof GemmXProof>;
  type GemmXProofJSON = ReturnType<InstanceType<typeof GemmXProof>["toJSON"]>;

  async function compileGemmXWithCache() {
    const cache = Cache.FileSystem(`./o1js-cache/GemmX-${name}`);
    return GemmX.compile({cache: cache});
  }

  async function gemmXBase(name: string, posId: number, layerId: number, rowId: number, ind: number, vkGemmX: VerificationKey) {
    let xs: Int64[][] = [];
    const bufX = await readBinary(`${zkDataDir}/pos_${posId}/layer_${layerId}/${name}_x.bin`);
    const xData = bufferToInt64ArrayLE(bufX);
    for(let i = 0; i < xData.length; i += InDim) {
      xs.push(xData.slice(i, i + InDim));
    }

    console.log(`${nowPrefix()} rowId = ${rowId}, ind = ${ind}: Finish reading x.`)

    let zStr: string = await readFile('proofs/embed/hash');
    let z = Field(zStr);
    let zk = fastPow(z, InDim);
    let zn = fastPow(z, OutDim);

    for(let j = ind; j < ind + 32; j++) {
      let rowStart = rowId;
      let rowEnd = rowStart + 1;
      let zkRowStart = fastPow(zk, rowId);
      let zkRowEnd = zkRowStart.mul(zk);
      let znRowStart = fastPow(zn, rowId);
      let znRowEnd = znRowStart.mul(zn);

      let left = j * ShortDim;
      let right = (j + 1) * ShortDim;
      let zLeft = fastPow(z, left);
      let zRight = fastPow(z, right);

      let input = new GemmXInput({
        rowStart: UInt32.from(rowStart),
        rowEnd: UInt32.from(rowEnd),
        zkRowStart,
        zkRowEnd,
        znRowStart,
        znRowEnd,

        left: UInt32.from(left),
        right: UInt32.from(right),
        zLeft,
        zRight,

        z,
        // zn,
      })

      let xArr = xs[rowId].slice(left, right);
      let xFieldArr = xArr.map((x) => x.toField())
      // console.log(`xArr.length: ${xArr.length}, xArr[1]: ${xArr[1]}, left: ${left}, right: ${right}, zLeft: ${zLeft}, zRight: ${zRight}`)
      const proof = await GemmX.base(input, xFieldArr);

      const start = performance.now();
      const ok = await verify(proof.proof, vkGemmX);
      const end = performance.now();

      let proofStr = JSON.stringify(proof.proof.toJSON());
      fs.mkdir(`proofs/pos_${posId}/layer_${layerId}/${name}/row_${rowId}`, { recursive: true }).then(() =>
          fs.writeFile(`proofs/pos_${posId}/layer_${layerId}/${name}/row_${rowId}/base_${ShortDimCount-1 + j}.json`, proofStr, "utf8")
        ).catch(console.error);

      console.log(`${nowPrefix()} ${name} GemmX row_${rowId} base proof ${ShortDimCount-1 + j} verify result: ${ok}, verifying time: ${end - start} ms`);
      // console.log(await NormX.analyzeMethods());
    }
  }

  async function gemmXMergeRow(name: string, tokenListLen: number, posId: number, layerId: number, ind: number, vkGemmX: VerificationKey) {
    let rowsProofs: GemmXProofType[] = new Array(tokenListLen);
    for (let i = 0; i < tokenListLen; i++) {
        let proofFile = `proofs/pos_${posId}/layer_${layerId}/${name}/row_${i}/base_${ind}.json`;
        let proofStr = await readFile(proofFile);
        const proofJson = JSON.parse(proofStr) as GemmXProofJSON;
        const proof = await GemmXProof.fromJSON(proofJson);
        // const ok = await verify(proof, vkNorm);
        // assert(ok, `base proof base_${rowId}_${j} verify failed!`);
        rowsProofs[i] = proof;
        // console.log(`add base_${ShortDimCount-1 + j}.json to proofs ${ShortDimCount - 1 + j}`);
    }

    console.log(`${nowPrefix()} ${name} Finish loading base proofs for ind ${ind}.`);

    while(rowsProofs.length > 1) {
      let rowsProofs2: GemmXProofType[]  = new Array(Math.floor((rowsProofs.length + 1) / 2));
      if(rowsProofs.length % 2 == 1) {
        rowsProofs2[rowsProofs2.length - 1] = rowsProofs[rowsProofs.length - 1];
        console.log(`${nowPrefix()} add proofs2 ${rowsProofs2.length - 1} directly`);
      }

      for(let i = 0; i < Math.floor(rowsProofs.length / 2); i++) {
        let upProof = rowsProofs[2 * i];
        let downProof = rowsProofs[2 * i + 1];
        let rowsInput = new GemmXInput({
          rowStart: upProof.publicInput.rowStart,
          rowEnd: downProof.publicInput.rowEnd,
          zkRowStart: upProof.publicInput.zkRowStart,
          zkRowEnd: downProof.publicInput.zkRowEnd,
          znRowStart: upProof.publicInput.znRowStart,
          znRowEnd: downProof.publicInput.znRowEnd,

          left: upProof.publicInput.left,
          right: upProof.publicInput.right,
          zLeft: upProof.publicInput.zLeft,
          zRight: upProof.publicInput.zRight,

          z: upProof.publicInput.z,
          // zn: upProof.publicInput.zn,
        })

        const proof = await GemmX.merge(rowsInput, upProof, downProof);

        const start = performance.now();
        const ok = await verify(proof.proof, vkGemmX);
        const end = performance.now();

        console.log(`${nowPrefix()} ${name} ind ${ind}: gemmXMergeRow merge proof ${i} verify result: ${ok}, verifying time: ${end - start} ms`);
        rowsProofs2[i] = proof.proof;
      }

      rowsProofs = rowsProofs2;
      // console.log('nodes length: ' + nodes.length);
    }

    const proof = rowsProofs[0];
    const ok = await verify(proof, vkGemmX);
    console.log(`${nowPrefix()} ${name} gemmXMergeRow merge proof verify result: ${ok}`);

    let proofStr = JSON.stringify(proof.toJSON());
    fs.mkdir(`proofs/pos_${posId}/layer_${layerId}/${name}/summary`, { recursive: true }).then(() =>
      fs.writeFile(`proofs/pos_${posId}/layer_${layerId}/${name}/summary/merge_${ind}.json`, proofStr, "utf8")
    ).catch(console.error);
  }

  // ++++++++++++

  class GemmWInput extends Struct({
    rowStart: UInt32,
    rowEnd: UInt32,
    zRowStart: Field,
    zRowEnd: Field,

    left: UInt32,
    right: UInt32,

    z: Field,
  }) {}

  class GemmWOutput extends Struct({
    wHash: Field,
    ZWs: ZSumArr,
  }) {}

  const GemmW = ZkProgram({
    name: 'GemmW',
    publicInput: GemmWInput,
    publicOutput: GemmWOutput,
    methods: {
      base: {
        privateInputs: [FieldArr,],
        // privateInputs: [RowSectionInt64],
        async method(input: GemmWInput, ws: Field[]) {
          input.rowEnd.assertEquals(input.rowStart.add(1));
          input.zRowEnd.assertEquals(input.zRowStart.mul(input.z), 'zRowStart zRowEnd not match');
          input.right.assertEquals(input.left.add(ShortDim));

          let wHash = Poseidon.hashPacked(FieldArr, ws);
          let ZWs: Field[] = new Array(ShortDim);

          for(let i = 0; i < ShortDim; i++) {
            // ZWs[i] = input.zRowStart.mul(asSigned32(ws[i]));  // Z^n * w[i]
            ZWs[i] = input.zRowStart.mul(ws[i]);  // Z^n * w[i]
          }

          const out = new GemmWOutput({
            wHash,
            ZWs,
          });
          return {publicOutput: out};
        },
      },
      merge: {
        privateInputs: [SelfProof, SelfProof],
        async method(input: GemmWInput,
          upProof: InstanceType<typeof SelfProof<GemmWInput, GemmWOutput> > ,
          downProof: InstanceType<typeof SelfProof<GemmWInput, GemmWOutput> >) {
            upProof.verify();
            downProof.verify();

            let upInput = upProof.publicInput;
            let upOutput = upProof.publicOutput;
            let downInput = downProof.publicInput;
            let downOutput = downProof.publicOutput;

            input.rowStart.assertEquals(upInput.rowStart);
            input.rowEnd.assertEquals(downInput.rowEnd);
            upInput.rowEnd.assertEquals(downInput.rowStart);

            input.zRowStart.assertEquals(upInput.zRowStart, 'zRowStart not equal');
            input.zRowEnd.assertEquals(downInput.zRowEnd, 'zRowEnd not equal');
            upInput.zRowEnd.assertEquals(downInput.zRowStart, 'zRowStart zRowEnd not equal');

            input.left.assertEquals(upInput.left);
            input.left.assertEquals(downInput.left);
            input.right.assertEquals(upInput.right);
            input.right.assertEquals(downInput.right);

            input.z.assertEquals(upInput.z, 'z 1 not equal');
            input.z.assertEquals(downInput.z, 'z 2 not equal');

            let wHash = Poseidon.hash([upOutput.wHash, downOutput.wHash]);
            let ZWs: Field[] = new Array(ShortDim);
            for(let i = 0; i < ShortDim; i++) {
              ZWs[i] = upOutput.ZWs[i].add(downOutput.ZWs[i]);
            }

            const out = new GemmWOutput({
              wHash,
              ZWs,
            });
            return {publicOutput: out};
        },
      },
    },
  });

  const GemmWProof = ZkProgram.Proof(GemmW);
  type GemmWProofType = InstanceType<typeof GemmWProof>;
  type GemmWProofJSON = ReturnType<InstanceType<typeof GemmWProof>["toJSON"]>;

  async function compileGemmWWithCache() {
    const cache = Cache.FileSystem(`./o1js-cache/GemmW-${name}`);
    return GemmW.compile({cache: cache});
  }

  async function gemmWBase(name: string, tokenListLen: number, posId: number, layerId: number, rowId: number, ind: number, vkGemmW: VerificationKey) {
    let ws: UInt32[][] = [];
    const bufX = await readBinary(`${zkDataDir}/pos_${posId}/layer_${layerId}/wq_a_w.bin`);
    const wData = bufferToUInt32ArrayLE(bufX);
    for(let i = 0; i < wData.length; i += InDim) {
      ws.push(wData.slice(i, i + InDim));
    }

    let zStr: string = await readFile('proofs/embed/hash');
    let z = Field(zStr);

    for(let j = ind; j < ind + 32; j++) {
      let rowStart = rowId;
      let rowEnd = rowStart + 1;
      let zRowStart = fastPow(z, rowId);
      let zRowEnd = zRowStart.mul(z);

      let left = j * ShortDim;
      let right = (j + 1) * ShortDim;

      let input = new GemmWInput({
        rowStart: UInt32.from(rowStart),
        rowEnd: UInt32.from(rowEnd),
        zRowStart,
        zRowEnd,

        left: UInt32.from(left),
        right: UInt32.from(right),

        z,
      })

      let wArr = ws[rowId].slice(left, right);
      let wFieldArr = wArr.map((w) => asSigned32(w))
      // console.log(`${nowPrefix()} xArr.length: ${xArr.length}, xArr[1]: ${xArr[1]}, left: ${left}, right: ${right}, zLeft: ${zLeft}, zRight: ${zRight}`)
      const proof = await GemmW.base(input, wFieldArr);
      // const proof = await NormX.base(input, xArr, wArr, qArr);

      const start = performance.now();
      const ok = await verify(proof.proof, vkGemmW);
      const end = performance.now();

      let proofStr = JSON.stringify(proof.proof.toJSON());
      fs.mkdir(`proofs/pos_${posId}/layer_${layerId}/${name}/row_${rowId}`, { recursive: true }).then(() =>
          fs.writeFile(`proofs/pos_${posId}/layer_${layerId}/${name}/row_${rowId}/w_base_${ShortDimCount-1 + j}.json`, proofStr, "utf8")
        ).catch(console.error);

      console.log(`${nowPrefix()} ${name} GemmW row_${rowId} w base proof ${ShortDimCount-1 + j} verify result: ${ok}, verifying time: ${end - start} ms`);
      // console.log(await NormX.analyzeMethods());
    }
  }

  async function getProof(path: string) {
    let proofStr = await readFile(path);
    const proofJson = JSON.parse(proofStr) as GemmWProofJSON;
    const proof = await GemmWProof.fromJSON(proofJson);
    return proof;
  }

  async function gemmWMergeRow(name: string, tokenListLen: number, posId: number, layerId: number, ind: number, rowIndex: number, vkGemmW: VerificationKey) {
    console.log(`${nowPrefix()} ind: ${ind}, Do gemmWMergeRow from ${rowIndex} to ${rowIndex + 31}`);

    for(let i = rowIndex; i < OutDim && i < rowIndex + 32; i++) {
      let leftProofInd = i - 1;
      let rightProofInd = i;

      let leftProofFile = `proofs/pos_${posId}/layer_${layerId}/${name}/weight/${ind}/w_merge_${leftProofInd}.json`;
      if(leftProofInd == 0) {
        leftProofFile = `proofs/pos_${posId}/layer_${layerId}/${name}/row_${leftProofInd}/w_base_${ind}.json`;
      }

      let rightProofFile = `proofs/pos_${posId}/layer_${layerId}/${name}/row_${rightProofInd}/w_base_${ind}.json`;

      let leftProof = await getProof(leftProofFile);
      let rightProof = await getProof(rightProofFile);

      let rowsInput = new GemmWInput({
        rowStart: leftProof.publicInput.rowStart,
        rowEnd: rightProof.publicInput.rowEnd,
        zRowStart: leftProof.publicInput.zRowStart,
        zRowEnd: rightProof.publicInput.zRowEnd,

        left: leftProof.publicInput.left,
        right: leftProof.publicInput.right,

        z: leftProof.publicInput.z,
      })

      const proof = await GemmW.merge(rowsInput, leftProof, rightProof);

      const start = performance.now();
      const ok = await verify(proof.proof, vkGemmW);
      const end = performance.now();

      console.log(`${nowPrefix()} ${name} ind ${ind}, i: ${i}: GemmWRows merge proof ${i} verify result: ${ok}, verifying time: ${end - start} ms`);

      let proofStr = JSON.stringify(proof.proof.toJSON())
      await fs.mkdir(`proofs/pos_${posId}/layer_${layerId}/${name}/weight/${ind}`, { recursive: true }).then(async () =>
        await fs.writeFile(`proofs/pos_${posId}/layer_${layerId}/${name}/weight/${ind}/w_merge_${i}.json`, proofStr, "utf8")
      ).catch(console.error);
    }
  }

  class GemmXWInput extends Struct({
    rowCount: UInt32,

    left: UInt32,
    right: UInt32,
    zLeft: Field,
    zRight: Field,

    z: Field,
  }) {}

  class GemmXWOutput extends Struct({
    wHash: Field,
    zsumX: Field,
    zsumY: Field,
  }) {}

  const GemmXW = ZkProgram({
    name: 'GemmXW',
    publicInput: GemmXWInput,
    publicOutput: GemmXWOutput,
    methods: {
      base: {
        privateInputs: [GemmXProof, GemmWProof],
        async method(input: GemmXWInput, xProof: GemmXProofType, wProof: GemmWProofType) {
          xProof.verify();
          wProof.verify();

          const xInput = xProof.publicInput;
          const xOutput = xProof.publicOutput;
          const wInput = wProof.publicInput;
          const wOutput = wProof.publicOutput;

          xInput.rowStart.assertEquals(UInt32.from(0));
          xInput.rowEnd.assertEquals(input.rowCount);

          xInput.zkRowStart.assertEquals(1, 'zkRowStart not equal');
          xInput.znRowStart.assertEquals(1, 'znRowStart not equal');

          input.left.assertEquals(xInput.left);
          input.right.assertEquals(xInput.right);
          input.zLeft.assertEquals(xInput.zLeft, 'zLeft not equal');
          input.zRight.assertEquals(xInput.zRight, 'zRight not equal');

          input.z.assertEquals(xInput.z, 'z not equal');

          wInput.rowStart.assertEquals(UInt32.from(0));
          wInput.rowEnd.assertEquals(UInt32.from(OutDim));

          wInput.zRowStart.assertEquals(1, 'w zRowStart not equal');

          input.left.assertEquals(wInput.left);
          input.right.assertEquals(wInput.right);

          input.z.assertEquals(wInput.z, 'z 2 not equal');

          let zsumY = Field(0);
          for(let i = 0; i < ShortDim; i++) {
            let dy = xOutput.ZXs[i].mul(wOutput.ZWs[i]);
            zsumY = zsumY.add(dy);
          }

          const out = new GemmXWOutput({
            wHash: wOutput.wHash,
            zsumX: xOutput.zsumX,
            zsumY,
          });
          return {publicOutput: out};
        }
      },

      merge: {
        privateInputs: [SelfProof, SelfProof],
        async method(input: GemmXWInput, leftProof: InstanceType<typeof SelfProof<GemmXWInput, GemmXWOutput> >,
          rightProof: InstanceType<typeof SelfProof<GemmXWInput, GemmXWOutput> >) {
            leftProof.verify();
            rightProof.verify();

            let leftInput = leftProof.publicInput;
            let leftOutput = leftProof.publicOutput;
            let rightInput = rightProof.publicInput;
            let rightOutput = rightProof.publicOutput;

            input.rowCount.assertEquals(leftInput.rowCount);
            input.rowCount.assertEquals(rightInput.rowCount);

            input.left.assertEquals(leftInput.left);
            input.right.assertEquals(rightInput.right);
            leftInput.right.assertEquals(rightInput.left);

            input.zLeft.assertEquals(leftInput.zLeft, 'zLeft not equal');
            input.zRight.assertEquals(rightInput.zRight, 'zRight not equal');
            leftInput.zRight.assertEquals(rightInput.zLeft, 'zLeft zRight not equal');

            input.z.assertEquals(leftInput.z, 'z 1 not equal');
            input.z.assertEquals(rightInput.z, 'z 2 not equal');

            const wHash = Poseidon.hash([leftOutput.wHash, rightOutput.wHash]);
            const zsumX = leftOutput.zsumX.add(rightOutput.zsumX);
            const zsumY = leftOutput.zsumY.add(rightOutput.zsumY);

            const out = new GemmXWOutput({
              wHash,
              zsumX,
              zsumY,
            });
            return {publicOutput: out};
      }
      }
    }
  })

  const GemmXWProof = ZkProgram.Proof(GemmXW);
  type GemmXWProofType = InstanceType<typeof GemmXWProof>;
  type GemmXWProofJSON = ReturnType<InstanceType<typeof GemmXWProof>["toJSON"]>;

  async function compileGemmXWWithCache() {
    const cache = Cache.FileSystem(`./o1js-cache/GemmXW-${name}`);
    await compileGemmXWithCache();
    await compileGemmWWithCache();
    return GemmXW.compile({cache: cache});
  }

  async function gemmXWBase(name: string, tokenListLen: number, posId: number, layerId: number, ind: number, vkGemmXW: VerificationKey) {
    let zStr: string = await readFile('proofs/embed/hash');
    let z = Field(zStr);

    for(let j = ind; j < ind + 32; j++) {
      let left = j * ShortDim;
      let right = (j + 1) * ShortDim;

      let input = new GemmXWInput({
        rowCount: UInt32.from(tokenListLen),
        left: UInt32.from(left),
        right: UInt32.from(right),
        zLeft: fastPow(z, left),
        zRight: fastPow(z, right),
        z,
      })

      let xProofFile = `proofs/pos_${posId}/layer_${layerId}/${name}/summary/merge_${ShortDimCount-1 + j}.json`;
      let xProofStr = await readFile(xProofFile);
      const xproofJson = JSON.parse(xProofStr) as GemmXProofJSON;
      const xProof = await GemmXProof.fromJSON(xproofJson);

      let wProofFile = `proofs/pos_${posId}/layer_${layerId}/${name}/weight/${ShortDimCount-1 + j}/w_merge_${OutDim - 1}.json`;
      let wProofStr = await readFile(wProofFile);
      const wproofJson = JSON.parse(wProofStr) as GemmWProofJSON;
      const wProof = await GemmWProof.fromJSON(wproofJson);

      const proof = await GemmXW.base(input, xProof, wProof);

      const start = performance.now();
      const ok = await verify(proof.proof, vkGemmXW);
      const end = performance.now();

      let proofStr = JSON.stringify(proof.proof.toJSON());
      fs.mkdir(`proofs/pos_${posId}/layer_${layerId}/${name}/summary`, { recursive: true }).then(() =>
          fs.writeFile(`proofs/pos_${posId}/layer_${layerId}/${name}/summary/xw_${ShortDimCount-1 + j}.json`, proofStr, "utf8")
        ).catch(console.error);

      console.log(`${nowPrefix()} ${name} GemmXW base proof ${ShortDimCount-1 + j} verify result: ${ok}, verifying time: ${end - start} ms`);
      // console.log(await NormX.analyzeMethods());
    }
  }

  async function gemmXWMerge(name: string, tokenListLen: number, posId: number, layerId: number, ind: number, vkGemmXW: VerificationKey) {
    let proofs: GemmXWProofType[] = new Array(2 * ShortDimCount - 1);
    for (let j = 0; j < ShortDimCount; j++) {
        let proofFile = `proofs/pos_${posId}/layer_${layerId}/${name}/summary/xw_${ShortDimCount-1 + j}.json`;
        let proofStr = await readFile(proofFile);
        const proofJson = JSON.parse(proofStr) as GemmXWProofJSON;
        const proof = await GemmXWProof.fromJSON(proofJson);
        // const ok = await verify(proof, vkNorm);
        // assert(ok, `base proof base_${rowId}_${j} verify failed!`);
        proofs[ShortDimCount - 1 + j] = proof;
        // console.log(`${nowPrefix()} add base_${ShortDimCount-1 + j}.json to proofs ${ShortDimCount - 1 + j}`);
    }

    for(let j = ind+1; j < ShortDimCount - 1; j++) {
      let proofFile = `proofs/pos_${posId}/layer_${layerId}/${name}/summary/xw_merge_${j}.json`;
      let proofStr = await readFile(proofFile);
      const proofJson = JSON.parse(proofStr) as GemmXWProofJSON;
      const proof = await GemmXWProof.fromJSON(proofJson);
      // const ok = await verify(proof, vkNorm);
      // assert(ok, `merge proof merge_${rowId}_${j} verify failed!`);
      proofs[j] = proof;
      // console.log(`${nowPrefix()} add merge_${j}.json to proofs ${j}`);
    }

    let count = 8;
    for(let j = ind; j >= 0; j--) {
      if(count == 0) break;
      count--;

      let leftProof = proofs[2 * j + 1];
      let rightProof = proofs[2 * j + 2];
      let gemmXWInput = new GemmXWInput({
        rowCount: leftProof.publicInput.rowCount,
        left: leftProof.publicInput.left,
        right: rightProof.publicInput.right,
        zLeft: leftProof.publicInput.zLeft,
        zRight: rightProof.publicInput.zRight,
        z: leftProof.publicInput.z,
      })

      // console.log(leftProof.publicInput.right);
      // console.log(rightProof.publicInput.left);

      const proof = await GemmXW.merge(gemmXWInput, leftProof, rightProof);

      const start = performance.now();
      const ok = await verify(proof.proof, vkGemmXW);
      const end = performance.now();

      console.log(`${nowPrefix()} ${name} GemmXW merge proof ${j} verify result: ${ok}, verifying time: ${end - start} ms`);
      proofs[j] = proof.proof;

      let proofStr = JSON.stringify(proof.proof.toJSON())
      fs.mkdir(`proofs/pos_${posId}/layer_${layerId}/${name}/summary`, { recursive: true }).then(() =>
        fs.writeFile(`proofs/pos_${posId}/layer_${layerId}/${name}/summary/xw_merge_${j}.json`, proofStr, "utf8")
      ).catch(console.error);
    }
  }

  async function checkGemm() {
    const InDim = 7168;
    const OutDim = 1536;

    let zStr: string = await readFile('proofs/embed/hash');
    let z = Field(zStr);

    let xs: Int64[][] = [];
    const bufX = await readBinary(`${zkDataDir}/pos_0/layer_0/wq_a_x.bin`);
    const xData = bufferToInt64ArrayLE(bufX);
    for(let i = 0; i < xData.length; i += InDim) {
      xs.push(xData.slice(i, i + InDim));
    }

    let zxs: Field[] = [];

    let zd: Field = fastPow(z, OutDim);
    for(let j = 0; j < InDim; j++) {
      let sum = Field(0);
      let zi = Field(1);
      for(let i = 0; i < 24; i++) {
        sum = sum.add(zi.mul(xs[i][j].toField()));
        zi = zi.mul(zd);
      }
      zxs.push(sum);
    }

    let ws: UInt32[][] = [];
    const bufW = await readBinary(`${zkDataDir}/pos_0/layer_0/wq_a_w.bin`);
    const wData = bufferToUInt32ArrayLE(bufW);
    for(let i = 0; i < wData.length; i += InDim) {
      ws.push(wData.slice(i, i + InDim));
    }

    let zws: Field[] = [];
    for(let i = 0; i < ShortDimCount; i++) {
      let wProofFile = `proofs/pos_0/layer_0/wq_a/weight/${ShortDimCount - 1 + i}/w_merge_1535.json`;
      let wProofStr = await readFile(wProofFile);
      const wproofJson = JSON.parse(wProofStr) as ReturnType<InstanceType<typeof GemmWProof>["toJSON"]>;
      const wProof = await GemmWProof.fromJSON(wproofJson);

      for(let j = 0; j < ShortDim; j++) {
        zws.push(wProof.publicOutput.ZWs[j]);
      }
      console.log(`${nowPrefix()} zw[${i * ShortDim}] = ${zws[i * ShortDim]}`);
    }

    let zis: Field[] = [];
    let zz = Field(1);
    for(let i = 0; i < OutDim; i++) {
      zis.push(zz);
      zz = zz.mul(z);
    }

    let zmul = Field(0);
    for(let j = 0; j < InDim; j++) {
      zmul = zmul.add(zxs[j].mul(zws[j]));
    }
    console.log(`${nowPrefix()} zmul: ${zmul}`);

    let qs: Int64[][] = [];
    const bufQ = await readBinary(`${zkDataDir}/pos_0/layer_0/q_norm_x.bin`);
    const qData = bufferToInt64ArrayLE(bufQ);
    for(let i = 0; i < qData.length; i += OutDim) {
      qs.push(qData.slice(i, i + OutDim));
    }

    let rs: Int64[][] = [];
    const bufPrevR = await readBinary(`${zkDataDir}/pos_0/layer_0/q_norm_r.bin`);
    const rData = bufferToInt64ArrayLE(bufPrevR);
    for(let i = 0; i < rData.length; i += OutDim) {
      rs.push(rData.slice(i, i + OutDim));
    }

    // let ys: Int64[][] = [];
    let zmul2 = Field(0);
    let zzz = Field(1);
    for(let i = 0; i < 24; i++) {
      // console.log(`${nowPrefix()} Begin i ${i}`);
      let yy: Int64[] = [];
      for(let j = 0; j < OutDim; j++) {
        let y2 = qs[i][j].mul(Int64.from(1n << 30n)).add(rs[i][j]);

        zmul2 = zmul2.add(zzz.mul(y2.toField()));
        zzz = zzz.mul(z);
      }
      // ys.push(yy);
    }
    console.log(`${nowPrefix()} zmul2: ${zmul2}`);
  }

  return { GemmX, GemmXProof, compileGemmXWithCache, gemmXBase, gemmXMergeRow,
          GemmW, GemmWProof, compileGemmWWithCache, gemmWBase, gemmWMergeRow,
          GemmXW, GemmXWProof, compileGemmXWWithCache, gemmXWBase, gemmXWMerge,
          checkGemm };
}

// End Gemm ===========================================================================================

// Begin Rope ===========================================================================================

function createRopeClass(name: string) {
  const n_local_heads = 128;
  const qk_rope_head_dim = 64;
  const ropeDim = n_local_heads * qk_rope_head_dim;

  const FieldArr = Provable.Array(Field, qk_rope_head_dim);
  const RowSectionInt64 = Provable.Array(Int64, qk_rope_head_dim);

  class RopeInput extends Struct({
    rowId: UInt32,

    headStart: UInt32,
    headEnd: UInt32,

    zStart: Field,
    zEnd: Field,

    z: Field,
  }) {}

  class RopeOutput extends Struct({
    wHash: Field,
    zsumX: Field,
    zsumY: Field,
  }) {}

  class QR extends Struct({ q: Int64, r: UInt64 }) {}

  const Rope = ZkProgram({
      name: 'Rope',
      publicInput: RopeInput,
      publicOutput: RopeOutput,
      methods: {
        base: {
          // x, w, q(商), r(余数)
          privateInputs: [RowSectionInt64, RowSectionInt64, RowSectionInt64, UInt64],
          // privateInputs: [RowSectionInt64, RowSectionInt32, RowSectionInt64],
          async method(input: RopeInput, xs: Int64[], ws: Int64[], ys: Int64[], rescale: UInt64) {
          // async method(input: NormXInput, xs: Int64[], ws: UInt32[], qs: Int64[]) {
            input.headEnd.assertEquals(input.headStart.add(1));

            let wHash = Poseidon.hashPacked(RowSectionInt64, ws);

            let z = input.z;
            let zi = input.zStart;
            let zsumX = Field(0);
            let zsumY = Field(0);
            for(let i = 0; i < qk_rope_head_dim; i+=2) {
              let a0 = xs[i];
              let a1 = xs[i+1];
              let b0 = ws[i];
              let b1 = ws[i+1];

              let a0F = a0.toField();
              let a1F = a1.toField();
              let b0F = b0.toField();
              let b1F = b1.toField();

              let aa0 = 0n, aa1 = 0n, bb0 = 0n, bb1 = 0n;
              Provable.asProver(() => {
                  aa0 = a0.toBigint();
                  aa1 = a1.toBigint();
                  bb0 = b0.toBigint();
                  bb1 = b1.toBigint();
              });

              let a0b0 = a0F.mul(b0F);
              let a1b1 = a1F.mul(b1F);
              let a0b1 = a0F.mul(b1F);
              let a1b0 = a1F.mul(b0F);

              const qr00 = Provable.witness(QR, () => {
                  let prd = aa0 * bb0;
                  let rr = rescale.toBigInt();
                  let q = prd / rr;
                  let r = prd % rr;
                  if(r < 0) {
                      r = r + rr;
                      q = q - BigInt(1);
                  }
                  // console.log(`${nowPrefix()} qr00: prd=${prd}, rr=${rr}, q=${q}, r: ${r}`);
                  return new QR({ q: Int64.from(q), r: UInt64.from(r) });
              });

              const qr11 = Provable.witness(QR, () => {
                  let prd = aa1 * bb1;
                  let rr = rescale.toBigInt();
                  let q = prd / rr;
                  let r = prd % rr;
                  if(r < 0) {
                      r = r + rr;
                      q = q - BigInt(1);
                  }
                  return new QR({ q: Int64.from(q), r: UInt64.from(r) });
              });
  
              const qr01 = Provable.witness(QR, () => {
                  let prd = aa0 * bb1;
                  let rr = rescale.toBigInt();
                  let q = prd / rr;
                  let r = prd % rr;
                  if(r < 0) {
                      r = r + rr;
                      q = q - BigInt(1);
                  }
                  return new QR({ q: Int64.from(q), r: UInt64.from(r) });
              });

              const qr10 = Provable.witness(QR, () => {
                  let prd = aa1 * bb0;
                  let rr = rescale.toBigInt();
                  let q = prd / rr;
                  let r = prd % rr;
                  if(r < 0) {
                      r = r + rr;
                      q = q - BigInt(1);
                  }
                  return new QR({ q: Int64.from(q), r: UInt64.from(r) });
              });

              // let rescaleF = rescale.toField();
              a0b0.assertEquals(qr00.q.toField().mul(rescale.value).add(qr00.r.value), `${i}: a0b0 != rescale * q + r`);
              // qr00.r.assertGreaterThanOrEqual(0, 'qr00.r not >= 0');
              qr00.r.assertLessThan(rescale, 'qr00.r not < rescale');

              a1b1.assertEquals(qr11.q.toField().mul(rescale.value).add(qr11.r.value),`${i}: a1b1 != rescale * q + r`);
              // qr11.r.assertGreaterThanOrEqual(0, 'qr11.r not >= 0');
              qr11.r.assertLessThan(rescale, 'qr11.r not < rescale');

              a0b1.assertEquals(qr01.q.toField().mul(rescale.value).add(qr01.r.value),`${i}: a0b1 != rescale * q + r`);
              // qr01.r.assertGreaterThanOrEqual(0, 'qr01.r not >= 0');
              qr01.r.assertLessThan(rescale, 'qr01.r not < rescale');

              a1b0.assertEquals(qr10.q.toField().mul(rescale.value).add(qr10.r.value),`${i}: a1b0 != rescale * q + r`);
              // qr10.r.assertGreaterThanOrEqual(0, 'qr10.r not >= 0');
              qr10.r.assertLessThan(rescale, 'qr10.r not < rescale');

              let y0 = ys[i];
              let y1 = ys[i+1];

              // Provable.asProver(() => console.log(`${nowPrefix()} 111 y0: ${y0}, a0: ${a0}, b0: ${b0}, qr00.q: ${qr00.q}, qr11.q: ${qr11.q}`));

              qr00.q.assertEquals(qr11.q.add(y0), `${i}: y0 != qr00.q - qr11.q`);
              y1.assertEquals(qr01.q.add(qr10.q), `${i}: y1 != qr01.q + qr10.q`);

              let dzx = zi.mul(a0F);
              let dzy = zi.mul(y0.toField());

              zi = zi.mul(z);

              dzx = dzx.add(zi.mul(a1F));
              dzy = dzy.add(zi.mul(y1.toField()));

              zsumX = zsumX.add(dzx);
              zsumY = zsumY.add(dzy);

              zi = zi.mul(z);
            }
            zi.assertEquals(input.zEnd, 'zEnd not match with zStart');

            const out = new RopeOutput({
              wHash: wHash,
              zsumX: zsumX,
              zsumY: zsumY,
            });
            return {publicOutput: out};
          },
        },
        merge: {
          privateInputs: [SelfProof, SelfProof],
          async method(input: RopeInput,
            leftProof: InstanceType<typeof SelfProof<RopeInput, RopeOutput> > ,
            rightProof: InstanceType<typeof SelfProof<RopeInput, RopeOutput> >) {
              leftProof.verify();
              rightProof.verify();

              let leftInput = leftProof.publicInput;
              let leftOutput = leftProof.publicOutput;
              let rightInput = rightProof.publicInput;
              let rightOutput = rightProof.publicOutput;

              input.rowId.assertEquals(leftInput.rowId);
              input.rowId.assertEquals(rightInput.rowId);

              input.headStart.assertEquals(leftInput.headStart);
              input.headEnd.assertEquals(rightInput.headEnd);
              leftInput.headEnd.assertEquals(rightInput.headStart);

              input.zStart.assertEquals(leftInput.zStart, 'zStart not equal');
              input.zEnd.assertEquals(rightInput.zEnd, 'zEnd not equal');
              leftInput.zEnd.assertEquals(rightInput.zStart, 'zStart zEnd not equal');

              input.z.assertEquals(leftInput.z, 'z 1 not equal');
              input.z.assertEquals(rightInput.z, 'z 2 not equal');

              leftOutput.wHash.assertEquals(rightOutput.wHash, 'wHash not equal')

              let zsumX = leftOutput.zsumX.add(rightOutput.zsumX);
              let zsumY = leftOutput.zsumY.add(rightOutput.zsumY);

              const out = new RopeOutput({
                wHash: leftOutput.wHash,
                zsumX: zsumX,
                zsumY: zsumY,
              });
              return {publicOutput: out};
          },
        },
      },
  });

  // const { verificationKey: vkRope } = await Rope.compile({cache: cache});
  const RopeProof = ZkProgram.Proof(Rope);
  type RopeProofType = InstanceType<typeof RopeProof>;
  type RopeProofJSON = ReturnType<InstanceType<typeof RopeProof>["toJSON"]>;

  async function compileRopeWithCache() {
      const cache = Cache.FileSystem(`./o1js-cache/Rope-${name}`);
      return Rope.compile({cache: cache});
  }

  async function calcRopeBase(name: string, posId: number, layerId: number, rowId: number, headId: number, vkRope: VerificationKey) {
      let xs: Int64[][][] = [];
      const bufX = await readBinary(`${zkDataDir}/pos_${posId}/layer_${layerId}/${name}_x.bin`);
      const xData = bufferToInt64ArrayLE(bufX);
      for(let i = 0; i < xData.length; i += ropeDim) {
          let xx: Int64[][] = [];
          for(let j = 0; j < ropeDim; j += qk_rope_head_dim) {
              xx.push(xData.slice(i + j, i + j + qk_rope_head_dim));
          }
          xs.push(xx);
      }

      let freqs: Int64[][] = [];
      const bufFreqs = await readBinary(`${zkDataDir}/freqs_cis.bin`);
      const freqsData = bufferToInt64ArrayLE(bufFreqs);
      for(let i = 0; i < freqsData.length; i += qk_rope_head_dim) {
          let d = freqsData.slice(i, i + qk_rope_head_dim);
          freqs.push(d);
      }

      let zStr: string = await readFile('proofs/embed/hash');
      let z = Field(zStr);

      let rescale =  1n << 42n;

      for(let i = headId; i < headId + 32; i++) {
          let ys: Int64[] = new Array(qk_rope_head_dim);
          for(let j = 0; j < qk_rope_head_dim; j+=2) {
              let a0 = xs[rowId][i][j].toBigint();
              let a1 = xs[rowId][i][j+1].toBigint();
              let b0 = freqs[rowId][j].toBigint();
              let b1 = freqs[rowId][j+1].toBigint();

              // console.log(`${nowPrefix()} a0: ${a0}, a1: ${a1}`);
              // console.log(`${nowPrefix()} b0: ${b0}, b1: ${b1}`);

              let a0b0 = a0 * b0;
              let a1b1 = a1 * b1;
              let a0b1 = a0 * b1;
              let a1b0 = a1 * b0;

              let q00 = a0b0 / rescale;
              let r00 = a0b0 % rescale;
              if(r00 < 0) {
                  q00 = q00 - 1n;
              }

              let q11 = a1b1 / rescale;
              let r11 = a1b1 % rescale;
              if(r11 < 0) {
                  q11 = q11 - 1n;
              }

              let q01 = a0b1 / rescale;
              let r01 = a0b1 % rescale;
              if(r01 < 0) {
                  q01 = q01 - 1n;
              }

              let q10 = a1b0 / rescale;
              let r10 = a1b0 % rescale;
              if(r10 < 0) {
                  q10 = q10 - 1n;
              }

              // ys[j] =   Int64.from(q00 - q11).toField();
              ys[j] =  Int64.from(q00 - q11);
              ys[j+1] = Int64.from(q01 + q10);

              // console.log(`${nowPrefix()} y[${j}]: ${ys[j]}, a0b0: ${q00}, a1b1: ${q11}`);
              // console.log(`${nowPrefix()} y[${j+1}]: ${ys[j+1]}, a0b1: ${q01}, a1b0: ${q10}`);
          }

          let headStart = i;
          let headEnd = i + 1;

          let zStart = fastPow(z, ropeDim * rowId + qk_rope_head_dim * i);
          let zEnd = fastPow(z, ropeDim * rowId + qk_rope_head_dim * (i+1));

          let input = new RopeInput({
              rowId: UInt32.from(rowId),

              headStart: UInt32.from(headStart),
              headEnd: UInt32.from(headEnd),

              zStart: zStart,
              zEnd: zEnd,

              z: z,
          })

          let xArr = xs[rowId][i];
          let wArr = freqs[rowId];

          // console.log(`${nowPrefix()} xArr.length: ${xArr.length}, wArr.length: ${wArr.length}, qArr.length: ${qArr.length}, rArr.length: ${rArr.length}`)
          const proof = await Rope.base(input, xArr, wArr, ys, UInt64.from(rescale));
          // const proof = await Rope.base(input, xs, freqs, ys, rescale);
          // const proof = await NormX.base(input, xArr, wArr, qArr);

          const start = performance.now();
          const ok = await verify(proof.proof, vkRope);
          const end = performance.now();

          let proofStr = JSON.stringify(proof.proof.toJSON());
          fs.mkdir(`proofs/pos_${posId}/layer_${layerId}/${name}_rope/row_${rowId}`, { recursive: true }).then(() =>
              fs.writeFile(`proofs/pos_${posId}/layer_${layerId}/${name}_rope/row_${rowId}/base_${n_local_heads - 1 + i}.json`, proofStr, "utf8")
              ).catch(console.error);
      
          console.log(`${nowPrefix()} ${name} Rope row_${rowId} base proof ${n_local_heads - 1 + i} verify result: ${ok}, verifying time: ${end - start} ms`);
      }
  }

  async function calcRopeMerge(name: string, posId: number, layerId: number, rowId: number, ind: number, vkRope: VerificationKey) {
      let proofs: RopeProofType[] = new Array(2 * n_local_heads - 1);
      for (let j = 0; j < n_local_heads; j++) {
          let proofFile = `proofs/pos_${posId}/layer_${layerId}/${name}_rope/row_${rowId}/base_${n_local_heads-1 + j}.json`;
          let proofStr = await readFile(proofFile);
          const proofJson = JSON.parse(proofStr) as RopeProofJSON;
          const proof = await RopeProof.fromJSON(proofJson);
          // const ok = await verify(proof, vkNorm); 
          // assert(ok, `base proof base_${rowId}_${j} verify failed!`);
          proofs[n_local_heads - 1 + j] = proof;
          // console.log(`${nowPrefix()} add base_${ShortDimCount-1 + j}.json to proofs ${ShortDimCount - 1 + j}`);
      }

      for(let j = ind+1; j < n_local_heads - 1; j++) {
        let proofFile = `proofs/pos_${posId}/layer_${layerId}/${name}_rope/row_${rowId}/merge_${j}.json`;
        let proofStr = await readFile(proofFile);
        const proofJson = JSON.parse(proofStr) as RopeProofJSON;
        const proof = await RopeProof.fromJSON(proofJson);
        // const ok = await verify(proof, vkNorm); 
        // assert(ok, `merge proof merge_${rowId}_${j} verify failed!`);
        proofs[j] = proof;
        // console.log(`${nowPrefix()} add merge_${j}.json to proofs ${j}`);
      }

      let count = 8;
      for(let j = ind; j >= 0; j--) {
        if(count == 0) break;
        count--;

        let leftProof = proofs[2 * j + 1];
        let rightProof = proofs[2 * j + 2];
        let ropeInput = new RopeInput({
          rowId: leftProof.publicInput.rowId,
          headStart: leftProof.publicInput.headStart,
          headEnd: rightProof.publicInput.headEnd,
          zStart: leftProof.publicInput.zStart,
          zEnd: rightProof.publicInput.zEnd,
          z: leftProof.publicInput.z,
        })

        // console.log(leftProof.publicInput.right);
        // console.log(rightProof.publicInput.left);

        const proof = await Rope.merge(ropeInput, leftProof, rightProof);

        const start = performance.now();
        const ok = await verify(proof.proof, vkRope);
        const end = performance.now();

        console.log(`${nowPrefix()} Rope row_${rowId} merge proof ${j} verify result: ${ok}, verifying time: ${end - start} ms`);
        proofs[j] = proof.proof;

        let proofStr = JSON.stringify(proof.proof.toJSON())
        await fs.writeFile(`proofs/pos_${posId}/layer_${layerId}/${name}_rope/row_${rowId}/merge_${j}.json`, proofStr, "utf8");
      }
    }

  class RopeRowsInput extends Struct({
      rowStart: UInt32,
      rowEnd: UInt32,

      zStart: Field,
      zEnd: Field,

      z: Field,
  }) {}

  class RopeRowsOutput extends Struct({
      wHash: Field,
      zsumX: Field,
      zsumY: Field,
  }) {}

  const ropeHashesLen = 24;

  const ropeHashes = [
      Field('0x10fec2754cd8b017c0d5c5806c143bd6f6244d1b939fd5a18dcd27a8c7191eb9'),
      Field('0x2fbd2a11774c166500dae5a6204d09d88f4115a0f7385d547687559a5e5cdb68'),
      Field('0x210d819af066aaf1a6bfc65f5f42ac49c5a020c6b0e6e23e537e498e8159d96b'),
      Field('0x238a55f7cd370ab7a29d2e17f5726c67ea2fb491781d4ce0398f7faa8d9bbd31'),
      Field('0x01ebe6445d6366bd4655e5da7e71588c9ecff47d6e1ea4aad85b12c52ccbef3f'),
      Field('0x1d29f775edfc20707ba94c45afbf5695cecf1b56c469591f168ec388fd59be83'),
      Field('0x1f87b303fbaa58c302b22ffe964e6b95467dfa0545be434be0b43f6ce6ff5258'),
      Field('0x2876dff53faf859d82e772e6d76103552fccb64cb39732250f59f904aeb47074'),
      Field('0x34909feb10aedacfece41ed44640eddacc067653d593a35e22cc771ef8b84887'),
      Field('0x28b5c4b924537d0434556c09bc6d81d014d97aa97ff0af855eb372bc22984663'),
      Field('0x24990bc3a7599cd625fce9ec69a8f81712f99dce57a71d573e9c4366dd24d800'),
      Field('0x1e20c9d9aecb606738d01b5423566fc6396e45b800244a3d93d2522777e603e2'),
      Field('0x1cd560e421b6b38fefdf38ba574117e8c6fd58fb12b6cfa247036b64d5cb2bfe'),
      Field('0x1ed4b8b8a25cdf1b732dcf64435f3924b88f3fc96f50edc64890e034ebb73fd1'),
      Field('0x0b94bcc60e9a70337a41fcb729d978cbda83c2847c9cb9ca7cc24c11ca46ab33'),
      Field('0x3b524a1721f3045756ea32300474c49744c626ec6fb1426094c27f62d9847dad'),
      Field('0x2c6b276edbbf587cb6c48a1857eb76ac3d4a8869dc129a6c37de6006ff068b0a'),
      Field('0x3f0cbcbf0a92720e8e06b2a1477816bf273a9d480f782bab21beb50bd10fba6e'),
      Field('0x0626ffc051feaf64e539710f15ee5d100e12e94996d44ac3df75dc7f66078963'),
      Field('0x1af7ac5c85e5c125ad600c12387af3c2ae6587b235e3f1a6e49344ffe1e9c262'),
      Field('0x097c45bad6e7355600d1648c4dc14a178a600feb75efe972cffa4070137c01db'),
      Field('0x2ffce1749192258189e2585488721d574c1724aab6f87db6adf2749ac12503e2'),
      Field('0x132cc9ebe88461387729b99144566d03fc0de8f18387ffe3021c89eff63c0021'),
      Field('0x19221f7c4824d03a22dcdb84c6697b0262c8e6fed2153b83d1ecad84831d8a8f'),
  ]

  function pickByIndex(idx: UInt32): Field {
      const mask = ropeHashes.map((_, i) => idx.equals(UInt32.from(i))); // Bool[]
      return Provable.switch(mask, Field, ropeHashes); // 只允许一个 true
    }

  const RopeRows = ZkProgram({
      name: 'RopeRows',
      publicInput: RopeRowsInput,
      publicOutput: RopeRowsOutput,
      methods: {
        base: {
          privateInputs: [RopeProof],
          async method(input: RopeRowsInput, proof: RopeProofType) {
            proof.verify();

            proof.publicInput.headStart.assertEquals(UInt32.from(0));
            proof.publicInput.headEnd.assertEquals(UInt32.from(n_local_heads));

            let rowId = proof.publicInput.rowId;

            input.rowStart.assertEquals(rowId);
            input.rowEnd.assertEquals(input.rowStart.add(1));

            let zn = fastPow(input.z, ropeDim);
            input.zStart.assertEquals(proof.publicInput.zStart, 'zStart not equal');
            input.zEnd.assertEquals(input.zStart.mul(zn), 'zEnd not equal');

            input.z.assertEquals(proof.publicInput.z, 'z not equal');

            let wHash = proof.publicOutput.wHash;

            // 比较 freqs的hash
            let h = pickByIndex(rowId);
            wHash.assertEquals(h, 'wHash not equal');

            const out = new RopeRowsOutput({
              wHash: wHash,
              zsumX: proof.publicOutput.zsumX,
              zsumY: proof.publicOutput.zsumY,
            });
            return {publicOutput: out};
          }
        },
        merge: {
          privateInputs: [SelfProof, SelfProof],
          async method(input: RopeRowsInput,
            upProof: InstanceType<typeof SelfProof<RopeRowsInput, RopeRowsOutput> > ,
            downProof: InstanceType<typeof SelfProof<RopeRowsInput, RopeRowsOutput> >) {
              upProof.verify();
              downProof.verify();

              let upInput = upProof.publicInput;
              let upOutput = upProof.publicOutput;
              let downInput = downProof.publicInput;
              let downOutput = downProof.publicOutput;

              input.rowStart.assertEquals(upInput.rowStart);
              input.rowEnd.assertEquals(downInput.rowEnd);
              upInput.rowEnd.assertEquals(downInput.rowStart)

              input.zStart.assertEquals(upInput.zStart, 'zStart not equal');
              input.zEnd.assertEquals(downInput.zEnd, 'zEnd not equal');
              upInput.zEnd.assertEquals(downInput.zStart, 'zStart zEnd not equal');

              input.z.assertEquals(upInput.z, 'z 1 not equal');
              input.z.assertEquals(downInput.z, 'z 2 not equal');

              // upOutput.wHash.assertEquals(downOutput.wHash, 'wHash not equal');
              let wHash = Poseidon.hash([upOutput.wHash, downOutput.wHash]);

              let zsumX = upOutput.zsumX.add(downOutput.zsumX);
              let zsumY = upOutput.zsumY.add(downOutput.zsumY);

              const out = new RopeRowsOutput({
                wHash: wHash,
                zsumX: zsumX,
                zsumY: zsumY,
              });
              return {publicOutput: out};
          },
        },
      }
    });

  const RopeRowsProof = ZkProgram.Proof(RopeRows);
  type RopeRowsProofType = InstanceType<typeof RopeRowsProof>;
  type RopeRowsProofJSON = ReturnType<InstanceType<typeof RopeRowsProof>["toJSON"]>;

  // let ropeRowsStr = await RopeRows.analyzeMethods();
  // console.log('RopeRows info: ', ropeRowsStr);

  async function compileRopeRowsWithCache() {
      const cache = Cache.FileSystem(`./o1js-cache/NormRows-${name}`);
  
      await compileRopeWithCache();
      return RopeRows.compile({cache: cache});
  }

  async function wrapRopeRow(tokenListLen: number, posId: number, layerId: number, vkRopeRows: VerificationKey) {
      for(let rowId = 0; rowId < tokenListLen; rowId++) {
        let proofFile = `proofs/pos_${posId}/layer_${layerId}/${name}_rope/row_${rowId}/merge_0.json`;
        let proofStr = await readFile(proofFile);
        const proofJson = JSON.parse(proofStr) as RopeProofJSON;
        const earlierProof = await RopeProof.fromJSON(proofJson);
        // const ok = await verify(proof, vkNorm); 
        // assert(ok, `merge proof merge_${rowId}_${j} verify failed!`);

        let publicInput = earlierProof.publicInput;
        let rId = publicInput.rowId;

        let input = new RopeRowsInput({
          rowStart: rId,
          rowEnd: rId.add(1),
          zStart: publicInput.zStart,
          zEnd: publicInput.zEnd,
          z: publicInput.z })

        const proof = await RopeRows.base(input, earlierProof);

        const start = performance.now();
        const ok = await verify(proof.proof, vkRopeRows);
        const end = performance.now();

        console.log(`${nowPrefix()} ${name} RopeRow base proof verify: ${ok}, verifying time: ${end - start} ms`);

        let proofResStr = JSON.stringify(proof.proof.toJSON());
        fs.mkdir(`proofs/pos_${posId}/layer_${layerId}/${name}_rope/summary`, { recursive: true }).then(() =>
          fs.writeFile(`proofs/pos_${posId}/layer_${layerId}/${name}_rope/summary/wrap_row_${rowId}.json`, proofResStr, "utf8")
        ).catch(console.error);

        console.log(`${nowPrefix()} ${name} RopeRows row_${rowId} wrap proof verify result: ${ok}`);
      }
    }

    async function mergeRopeRow(tokenListLen: number, posId: number, layerId: number, vkRopeRows: VerificationKey) {
      let proofFile0 = `proofs/pos_${posId}/layer_${layerId}/${name}_rope/summary/wrap_row_0.json`;
      let proofStr0 = await readFile(proofFile0);
      const proofJson0 = JSON.parse(proofStr0) as RopeRowsProofJSON;
      let prevProof = await RopeRowsProof.fromJSON(proofJson0);

      for (let i = 1; i < tokenListLen; i++) {
          let proofFile = `proofs/pos_${posId}/layer_${layerId}/${name}_rope/summary/wrap_row_${i}.json`;
          let proofStr = await readFile(proofFile);
          const proofJson = JSON.parse(proofStr) as RopeRowsProofJSON;
          const currProof = await RopeRowsProof.fromJSON(proofJson);

          let rowsInput = new RopeRowsInput({
              rowStart: prevProof.publicInput.rowStart,
              rowEnd: currProof.publicInput.rowEnd,
              zStart: prevProof.publicInput.zStart,
              zEnd: currProof.publicInput.zEnd,
              z: prevProof.publicInput.z,
          })

          const proof = await RopeRows.merge(rowsInput, prevProof, currProof);

          const start = performance.now();
          const ok = await verify(proof.proof, vkRopeRows);
          const end = performance.now();

          console.log(`${nowPrefix()} ${name} RopeRows merge proof ${i} verify result: ${ok}, verifying time: ${end - start} ms`);
          prevProof = proof.proof;
      }

      const proof = prevProof;
      const ok = await verify(proof, vkRopeRows);
      console.log(`${nowPrefix()} ${name} RopeRows merge proof verify result: ${ok}`);
  
      let proofStr = JSON.stringify(proof.toJSON())
      await fs.writeFile(`proofs/pos_${posId}/layer_${layerId}/${name}_rope/summary/rope.json`, proofStr, "utf8");
    }

  async function calcRopeHashes() {
      const posId = Number(process.argv[3]);
      const layerId = Number(process.argv[4]);

      let freqs: Int64[][] = [];
      const bufX = await readBinary(`${zkDataDir}/freqs_cis.bin`);
      const freqsData = bufferToInt64ArrayLE(bufX);
      for(let i = 0; i < freqsData.length; i += qk_rope_head_dim) {
          freqs.push(freqsData.slice(i, i + qk_rope_head_dim));
      }

      // let freqsField: Field[][] = freqs.map(xs => (xs.map(x => x.toField())));

      // console.log('freqsField.length: ', freqsField.length);
      for(let i = 0; i < freqs.length; i++) {
          let h = Poseidon.hashPacked(RowSectionInt64, freqs[i]);

          // let h = Poseidon.hash(freqsField[i]);
          let hexH: string = fieldToHex(h);
          console.log(hexH);
      }
  }

  return { Rope, RopeProof, compileRopeWithCache, calcRopeBase, calcRopeMerge,
    RopeRows, RopeRowsProof, compileRopeRowsWithCache, wrapRopeRow, mergeRopeRow,
    calcRopeHashes };
}

// End Rope ===========================================================================================

// Begin Softmax ===========================================================================================

function createSoftmaxClass(name: string) {
    const n_local_heads = 128;
    const SectionSize = 32;

    const LOG_TABLE_SIZE = 8;

    const RowSectionUInt64 = Provable.Array(UInt64, SectionSize);

    const TWO63 = 1n << 63n;
    const TWO19 = UInt64.from(1n << 19n);

    const LOG2E_Q19 = UInt64.from(756388n);

    // const FieldArr = Provable.Array(Field, qk_rope_head_dim);
    // const RowSectionInt64 = Provable.Array(Int64, qk_rope_head_dim);

    class SectionInput extends Struct({
        rowId: UInt32,

        left: UInt32,
        right: UInt32,
        zLeft: Field,
        zRight: Field,

        z: Field,

        xmax: UInt64,
        sum: UInt64,
    }) {}

    class SectionOutput extends Struct({
        sumW: UInt64,
        zsumX: Field,
        zsumY: Field,
    }) {}

    // class QR extends Struct({ q: Field, r: Field }) {}

    const q19Table = [
        UInt64.from(524288n), UInt64.from(522870n), UInt64.from(521456n), UInt64.from(520046n), UInt64.from(518640n), UInt64.from(517238n), UInt64.from(515839n), UInt64.from(514444n),
        UInt64.from(513053n), UInt64.from(511666n), UInt64.from(510282n), UInt64.from(508903n), UInt64.from(507526n), UInt64.from(506154n), UInt64.from(504786n), UInt64.from(503421n),
        UInt64.from(502059n), UInt64.from(500702n), UInt64.from(499348n), UInt64.from(497998n), UInt64.from(496651n), UInt64.from(495308n), UInt64.from(493969n), UInt64.from(492633n),
        UInt64.from(491301n), UInt64.from(489973n), UInt64.from(488648n), UInt64.from(487327n), UInt64.from(486009n), UInt64.from(484695n), UInt64.from(483384n), UInt64.from(482077n),
        UInt64.from(480774n), UInt64.from(479474n), UInt64.from(478177n), UInt64.from(476884n), UInt64.from(475595n), UInt64.from(474309n), UInt64.from(473026n), UInt64.from(471747n),
        UInt64.from(470472n), UInt64.from(469200n), UInt64.from(467931n), UInt64.from(466666n), UInt64.from(465404n), UInt64.from(464145n), UInt64.from(462890n), UInt64.from(461639n),
        UInt64.from(460390n), UInt64.from(459146n), UInt64.from(457904n), UInt64.from(456666n), UInt64.from(455431n), UInt64.from(454200n), UInt64.from(452972n), UInt64.from(451747n),
        UInt64.from(450525n), UInt64.from(449307n), UInt64.from(448092n), UInt64.from(446881n), UInt64.from(445672n), UInt64.from(444467n), UInt64.from(443265n), UInt64.from(442067n),

        UInt64.from(440871n), UInt64.from(439679n), UInt64.from(438490n), UInt64.from(437305n), UInt64.from(436122n), UInt64.from(434943n), UInt64.from(433767n), UInt64.from(432594n),
        UInt64.from(431424n), UInt64.from(430258n), UInt64.from(429094n), UInt64.from(427934n), UInt64.from(426777n), UInt64.from(425623n), UInt64.from(424472n), UInt64.from(423325n),
        UInt64.from(422180n), UInt64.from(421038n), UInt64.from(419900n), UInt64.from(418764n), UInt64.from(417632n), UInt64.from(416503n), UInt64.from(415377n), UInt64.from(414254n),
        UInt64.from(413133n), UInt64.from(412016n), UInt64.from(410902n), UInt64.from(409791n), UInt64.from(408683n), UInt64.from(407578n), UInt64.from(406476n), UInt64.from(405377n),
        UInt64.from(404281n), UInt64.from(403188n), UInt64.from(402097n), UInt64.from(401010n), UInt64.from(399926n), UInt64.from(398845n), UInt64.from(397766n), UInt64.from(396691n),
        UInt64.from(395618n), UInt64.from(394548n), UInt64.from(393481n), UInt64.from(392417n), UInt64.from(391356n), UInt64.from(390298n), UInt64.from(389243n), UInt64.from(388190n),
        UInt64.from(387141n), UInt64.from(386094n), UInt64.from(385050n), UInt64.from(384009n), UInt64.from(382970n), UInt64.from(381935n), UInt64.from(380902n), UInt64.from(379872n),
        UInt64.from(378845n), UInt64.from(377821n), UInt64.from(376799n), UInt64.from(375780n), UInt64.from(374764n), UInt64.from(373751n), UInt64.from(372740n), UInt64.from(371732n),

        UInt64.from(370727n), UInt64.from(369725n), UInt64.from(368725n), UInt64.from(367728n), UInt64.from(366734n), UInt64.from(365742n), UInt64.from(364753n), UInt64.from(363767n),
        UInt64.from(362783n), UInt64.from(361802n), UInt64.from(360824n), UInt64.from(359848n), UInt64.from(358875n), UInt64.from(357905n), UInt64.from(356937n), UInt64.from(355972n),
        UInt64.from(355009n), UInt64.from(354050n), UInt64.from(353092n), UInt64.from(352137n), UInt64.from(351185n), UInt64.from(350236n), UInt64.from(349289n), UInt64.from(348344n),
        UInt64.from(347402n), UInt64.from(346463n), UInt64.from(345526n), UInt64.from(344592n), UInt64.from(343660n), UInt64.from(342731n), UInt64.from(341804n), UInt64.from(340880n),
        UInt64.from(339958n), UInt64.from(339039n), UInt64.from(338122n), UInt64.from(337208n), UInt64.from(336296n), UInt64.from(335387n), UInt64.from(334480n), UInt64.from(333576n),
        UInt64.from(332674n), UInt64.from(331774n), UInt64.from(330877n), UInt64.from(329982n), UInt64.from(329090n), UInt64.from(328200n), UInt64.from(327313n), UInt64.from(326428n),
        UInt64.from(325545n), UInt64.from(324665n), UInt64.from(323787n), UInt64.from(322911n), UInt64.from(322038n), UInt64.from(321168n), UInt64.from(320299n), UInt64.from(319433n),
        UInt64.from(318569n), UInt64.from(317708n), UInt64.from(316849n), UInt64.from(315992n), UInt64.from(315138n), UInt64.from(314286n), UInt64.from(313436n), UInt64.from(312588n),

        UInt64.from(311743n), UInt64.from(310900n), UInt64.from(310059n), UInt64.from(309221n), UInt64.from(308385n), UInt64.from(307551n), UInt64.from(306719n), UInt64.from(305890n),
        UInt64.from(305063n), UInt64.from(304238n), UInt64.from(303415n), UInt64.from(302595n), UInt64.from(301777n), UInt64.from(300961n), UInt64.from(300147n), UInt64.from(299335n),
        UInt64.from(298526n), UInt64.from(297719n), UInt64.from(296914n), UInt64.from(296111n), UInt64.from(295310n), UInt64.from(294512n), UInt64.from(293716n), UInt64.from(292921n),
        UInt64.from(292129n), UInt64.from(291339n), UInt64.from(290552n), UInt64.from(289766n), UInt64.from(288982n), UInt64.from(288201n), UInt64.from(287422n), UInt64.from(286645n),
        UInt64.from(285870n), UInt64.from(285097n), UInt64.from(284326n), UInt64.from(283557n), UInt64.from(282790n), UInt64.from(282026n), UInt64.from(281263n), UInt64.from(280502n),
        UInt64.from(279744n), UInt64.from(278988n), UInt64.from(278233n), UInt64.from(277481n), UInt64.from(276731n), UInt64.from(275982n), UInt64.from(275236n), UInt64.from(274492n),
        UInt64.from(273750n), UInt64.from(273009n), UInt64.from(272271n), UInt64.from(271535n), UInt64.from(270801n), UInt64.from(270069n), UInt64.from(269338n), UInt64.from(268610n),
        UInt64.from(267884n), UInt64.from(267159n), UInt64.from(266437n), UInt64.from(265717n), UInt64.from(264998n), UInt64.from(264282n), UInt64.from(263567n), UInt64.from(262854n),
    ]

    // const q19TableUint64: UInt64[] = q19Table.map(x => UInt64.from(x))

    function pickByIndex(idx: UInt64): UInt64 {
        const mask = q19Table.map((_, i) => idx.equals(UInt64.from(i))); // Bool[]
        return Provable.switch(mask, UInt64, q19Table); // 只允许一个 true
    }

    function shrVar(x: UInt64, k: UInt64): UInt64 {
        // 取 k 的低 6 位作为移位量（0..63）
        const kbits = k.toBits(); // Bool[32]，我们只用前 6 位
        const b0 = kbits[0], b1 = kbits[1], b2 = kbits[2], b3 = kbits[3], b4 = kbits[4], b5 = kbits[5];

        // 逐级条件移位：1,2,4,8,16,32
        let v = x;
        const v1  = v.rightShift(1);  v = Provable.if(b0, v1, v);
        const v2  = v.rightShift(2);  v = Provable.if(b1, v2, v);
        const v4  = v.rightShift(4);  v = Provable.if(b2, v4, v);
        const v8  = v.rightShift(8);  v = Provable.if(b3, v8, v);
        const v16 = v.rightShift(16);  v = Provable.if(b4, v16, v);
        const v32 = v.rightShift(32);  v = Provable.if(b5, v32, v);

        return v; // 相当于 x >>> k（逻辑右移）
      }

    const Section = ZkProgram({
        name: 'Section',
        publicInput: SectionInput,
        publicOutput: SectionOutput,
        methods: {
            base: {
                privateInputs: [RowSectionUInt64],
                async method(input: SectionInput, xs: UInt64[]) {
                    let zsumX = Field(0);
                    let zsumY = Field(0);
                    let sumW = UInt64.from(0);
                    let xmax = input.xmax;

                    let zi = input.zLeft;
                    for(let i = 0; i < SectionSize; i++) {
                        let b = input.left.add(i).lessThan(input.right);

                        let x = xs[i];
                        x.assertLessThanOrEqual(Provable.if(b, xmax, x));

                        let d = xmax.sub(x);

                        let y = d.mul(LOG2E_Q19).rightShift(19);
                        let k = y.rightShift(19); // d.mul(LOG2E_Q19) 最大 2^63, k 最大 2^25
                        let f = y.sub(k.mul(TWO19)).rightShift(19 - LOG_TABLE_SIZE);
                        let t = pickByIndex(f);

                        let ww = shrVar(t, k);

                        let wi = Provable.if(k.greaterThanOrEqual(UInt64.from(64)), UInt64.from(0), ww);

                        wi = Provable.if(b, wi, UInt64.from(0));

                        // Provable.asProver(() => {
                        //     if(input.rowId.toBigInt() == 2n) {
                        //         console.log(`i: ${i}, k: ${k}, wi: ${wi}`);
                        //     }
                        // });

                        sumW = sumW.add(wi);

                        let num = wi.leftShift(19);
                        let c = Provable.if(input.sum.equals(UInt64.from(0)), UInt64.from(0), num.div(input.sum));

                        let czi = Provable.if(b, zi.mul(c.value), Field(0));
                        zsumY = zsumY.add(czi);

                        let xx = 0n;
                        Provable.asProver(() => {
                            xx = x.toBigInt();
                        });

                        const xraw = Provable.witness(Field, () => {
                            let x2 = xx - TWO63;
                            return Field(x2);
                        });
                        xraw.add(TWO63).assertEquals(x.value);

                        let xzi = Provable.if(b, zi.mul(xraw), Field(0));
                        zsumX = zsumX.add(xzi);

                        zi = Provable.if(b, zi.mul(input.z), zi);
                    };
                    zi.assertEquals(input.zRight, 'zLeft zRight not match');

                    const out = new SectionOutput({
                        sumW,
                        zsumX,
                        zsumY,
                      });
                    return {publicOutput: out};
                }
            },
            merge: {
                privateInputs: [SelfProof, SelfProof],
                async method(input: SectionInput,
                    leftProof: InstanceType<typeof SelfProof<SectionInput, SectionOutput> > ,
                    rightProof: InstanceType<typeof SelfProof<SectionInput, SectionOutput> >) {

                    leftProof.verify();
                    rightProof.verify();

                    let leftInput = leftProof.publicInput;
                    let leftOutput = leftProof.publicOutput;
                    let rightInput = rightProof.publicInput;
                    let rightOutput = rightProof.publicOutput;

                    input.left.assertEquals(leftInput.left);
                    input.right.assertEquals(rightInput.right);
                    leftInput.right.assertEquals(rightInput.left);

                    input.zLeft.assertEquals(leftInput.zLeft, 'zLeft not equal');
                    input.zRight.assertEquals(rightInput.zRight, 'zRight not equal');
                    leftInput.zRight.assertEquals(rightInput.zLeft, 'zLeft zRight not equal');

                    input.z.assertEquals(leftInput.z, 'z 1 not equal');
                    input.z.assertEquals(rightInput.z, 'z 2 not equal');

                    input.xmax.assertEquals(leftInput.xmax);
                    input.xmax.assertEquals(rightInput.xmax);

                    input.sum.assertEquals(leftInput.sum);
                    input.sum.assertEquals(rightInput.sum);

                    let sumW = leftOutput.sumW.add(rightOutput.sumW);
                    let zsumX = leftOutput.zsumX.add(rightOutput.zsumX);
                    let zsumY = leftOutput.zsumY.add(rightOutput.zsumY);

                    const out = new SectionOutput({
                        sumW,
                        zsumX,
                        zsumY,
                        });
                    return {publicOutput: out};
                }
            }
        }
    })

    const SectionProof = ZkProgram.Proof(Section);
    type SectionProofType = InstanceType<typeof SectionProof>;
    type SectioinProofJSON = ReturnType<InstanceType<typeof SectionProof>["toJSON"]>;

    async function compileSoftmaxSectionWithCache() {
        const cache = Cache.FileSystem(`./o1js-cache/Softmax-section-${name}`);

        // let sectionStr = await Section.analyzeMethods();
        // console.log('Section info: ', sectionStr);
    
        return Section.compile({cache: cache});
    }

    class HeadInput extends Struct({
        rowId: UInt32,

        headDim: UInt32,

        headStart: UInt32,
        headEnd: UInt32,

        zStart: Field,
        zEnd: Field,

        z: Field,
    }) {}

    class HeadOutput extends Struct({
        zsumX: Field,
        zsumY: Field,
    }) {}

    const Head = ZkProgram({
        name: 'Head',
        publicInput: HeadInput,
        publicOutput: HeadOutput,
        methods: {
            base: {
                privateInputs: [SectionProof],
                async method(input: HeadInput, secProof: SectionProofType) {
                    secProof.verify();

                    let prfInput = secProof.publicInput;
                    let prfOutput = secProof.publicOutput;

                    // Provable.asProver(() => {
                    //     console.log(`${prfInput.left.toBigInt()}, ${input.headStart.toBigInt()}, ${input.headDim.toBigInt()}`);
                    // });

                    input.rowId.assertEquals(prfInput.rowId);
                    input.headStart.mul(input.headDim).assertEquals(prfInput.left);
                    input.headEnd.assertEquals(input.headStart.add(1));

                    input.zStart.assertEquals(prfInput.zLeft, 'zStart not equal');
                    input.zEnd.assertEquals(prfInput.zRight, 'zEnd not euqal');

                    input.z.assertEquals(prfInput.z, 'z not equal');

                    prfInput.sum.assertEquals(prfOutput.sumW);

                    const out = new HeadOutput({
                        zsumX: prfOutput.zsumX,
                        zsumY: prfOutput.zsumY,
                        });
                    return {publicOutput: out};
                }
            },

            merge: {
                privateInputs: [SelfProof, SelfProof],
                async method(input: HeadInput,
                    leftProof: InstanceType<typeof SelfProof<HeadInput, HeadOutput> > ,
                    rightProof: InstanceType<typeof SelfProof<HeadInput, HeadOutput> >) {

                    leftProof.verify();
                    rightProof.verify();

                    let leftInput = leftProof.publicInput;
                    let leftOutput = leftProof.publicOutput;
                    let rightInput = rightProof.publicInput;
                    let rightOutput = rightProof.publicOutput;

                    input.rowId.assertEquals(leftInput.rowId);
                    input.rowId.assertEquals(rightInput.rowId);

                    input.headDim.assertEquals(leftInput.headDim);
                    input.headDim.assertEquals(rightInput.headDim);

                    input.headStart.assertEquals(leftInput.headStart);
                    input.headEnd.assertEquals(rightInput.headEnd);
                    leftInput.headEnd.assertEquals(rightInput.headStart);

                    input.zStart.assertEquals(leftInput.zStart, 'zStart not equal');
                    input.zEnd.assertEquals(rightInput.zEnd, 'zEnd not equal');
                    leftInput.zEnd.assertEquals(rightInput.zStart, 'zStart zEnd not equal');

                    input.z.assertEquals(leftInput.z, 'z 1 not equal');
                    input.z.assertEquals(rightInput.z, 'z 2 not equal');

                    let zsumX = leftOutput.zsumX.add(rightOutput.zsumX);
                    let zsumY = leftOutput.zsumY.add(rightOutput.zsumY);

                    const out = new HeadOutput({
                        zsumX,
                        zsumY,
                        });
                    return {publicOutput: out};
                }
            }
        }
    })

    const HeadProof = ZkProgram.Proof(Head);
    type HeadProofType = InstanceType<typeof HeadProof>;
    type HeadProofJSON = ReturnType<InstanceType<typeof HeadProof>["toJSON"]>;


    async function compileSoftmaxHeadWithCache() {
        const cache = Cache.FileSystem(`./o1js-cache/Softmax-head-${name}`);

        let { verificationKey: vkSection } =  await compileSoftmaxSectionWithCache();
        let { verificationKey: vkHead } =  await Head.compile({cache: cache});
        return {vkSection, vkHead}
    }

    class RowsInput extends Struct({
        rowStart: UInt32,
        rowEnd: UInt32,
        zRowStart: Field,
        zRowEnd: Field,

        headDim: UInt32,
        headCount: UInt32,

        z: Field,
    }) {}

    class RowsOutput extends Struct({
        zsumX: Field,
        zsumY: Field,
    }) {}

    const Rows = ZkProgram({
        name: 'Rows',
        publicInput: RowsInput,
        publicOutput: RowsOutput,
        methods: {
            base: {
                privateInputs: [HeadProof],
                async method(input: RowsInput, headProof: HeadProofType) {
                    headProof.verify();

                    let hInput = headProof.publicInput;
                    let hOutput = headProof.publicOutput;

                    input.rowStart.assertEquals(hInput.rowId);
                    input.rowEnd.assertEquals(input.rowStart.add(1));
                    input.zRowStart.assertEquals(hInput.zStart, 'zStart not equal');
                    input.zRowEnd.assertEquals(hInput.zEnd, 'zEnd not equal');

                    hInput.headStart.assertEquals(UInt32.from(0));
                    hInput.headEnd.assertEquals(input.headCount);

                    input.headDim.assertEquals(hInput.headDim);

                    input.z.assertEquals(hInput.z, 'z not equal');

                    const out = new RowsOutput({
                        zsumX: hOutput.zsumX,
                        zsumY: hOutput.zsumY,
                        });
                    return {publicOutput: out};
                }
            },

            merge: {
                privateInputs: [SelfProof, SelfProof],
                async method(input: RowsInput,
                    upProof: InstanceType<typeof SelfProof<RowsInput, RowsOutput> > ,
                    downProof: InstanceType<typeof SelfProof<RowsInput, RowsOutput> >) {

                    upProof.verify();
                    downProof.verify();

                    let upInput = upProof.publicInput;
                    let upOutput = upProof.publicOutput;
                    let downInput = downProof.publicInput;
                    let downOutput = downProof.publicOutput;

                    input.rowStart.assertEquals(upInput.rowStart);
                    input.rowEnd.assertEquals(downInput.rowEnd);
                    upInput.rowEnd.assertEquals(downInput.rowStart);

                    input.zRowStart.assertEquals(upInput.zRowStart, 'zRowStart not equal');
                    input.zRowEnd.assertEquals(downInput.zRowEnd, 'zRowEnd not equal');
                    upInput.zRowEnd.assertEquals(downInput.zRowStart, 'zRowStart zRowEnd not equal');

                    input.headDim.assertEquals(upInput.headDim);
                    input.headDim.assertEquals(downInput.headDim);

                    input.headCount.assertEquals(upInput.headCount);
                    input.headCount.assertEquals(downInput.headCount);

                    input.z.assertEquals(upInput.z, 'z 1 not equal');
                    input.z.assertEquals(downInput.z, 'z 2 not equal');

                    let zsumX = upOutput.zsumX.add(downOutput.zsumX);
                    let zsumY = upOutput.zsumY.add(downOutput.zsumY);

                    const out = new RowsOutput({
                        zsumX,
                        zsumY,
                        });
                    return {publicOutput: out};
                }
            }
        }
    })

    const RowsProof = ZkProgram.Proof(Rows);
    type RowsProofType = InstanceType<typeof RowsProof>;
    type RowsProofJSON = ReturnType<InstanceType<typeof RowsProof>["toJSON"]>;


    async function compileSoftmaxRowsWithCache() {
        const cache = Cache.FileSystem(`./o1js-cache/Softmax-rows-${name}`);

        let { verificationKey: vkSection } =  await compileSoftmaxSectionWithCache();
        let { verificationKey: vkHead } =  await Head.compile({cache: cache});
        return Rows.compile({cache: cache});
    }

    function calcSumW(xs: UInt64[]) {
        let sumW = UInt64.from(0);
        let xmax = xs[0];
        for(let i = 1; i < xs.length; i++) {
            if(xs[i] > xmax) {
                xmax = xs[i];
            }
        }

        for(let i = 0; i < xs.length; i++) {
            let x = xs[i];
            let d = xmax.sub(x);

            if (d.toBigInt() > (64n << 19n)) {
                continue;
            }

            let y = d.mul(UInt64.from(LOG2E_Q19)).rightShift(19);
            let k = y.rightShift(19);

            if(k.toBigInt() >= 64n) {
                continue;
            }

            let f = y.sub(k.mul(TWO19)).rightShift(19 - LOG_TABLE_SIZE);
            let t = pickByIndex(f);

            let wi = shrVar(t, k);

            // console.log(`i: ${i}, d: ${d}, y: ${y}, k: ${k}, f: ${f}, t: ${t}, wi: ${wi}`);

            sumW = sumW.add(wi);
        }

        return {xmax, sumW};
    }

    async function calcHeadBase(name: string, posId: number, layerId: number, rowId: number, headId: number, headDim: number,
                                    vkSection: VerificationKey, vkHead: VerificationKey) {
        let xs: Int64[][][] = [];
        const bufX = await readBinary(`${zkDataDir}/pos_${posId}/layer_${layerId}/${name}_softmax_x.bin`);
        const xData = bufferToInt64ArrayLE(bufX);
        for(let i = 0; i < xData.length; i += n_local_heads * headDim) {
            let xx: Int64[][] = [];
            for(let j = 0; j < n_local_heads * headDim; j += headDim) {
                xx.push(xData.slice(i + j, i + j + headDim));
            }
            xs.push(xx);
        }

        let zStr: string = await readFile('proofs/embed/hash');
        let z = Field(zStr);

        let SectionCount = Math.floor((headDim + SectionSize - 1) / SectionSize);

        // console.log(`SectionCount: ${SectionCount}`);

        let xx = xs[rowId][headId].map(x => UInt64.from(x.toBigint() + TWO63));
        let {xmax, sumW} = calcSumW(xx);
        // console.log(`xmax: ${xmax.toBigInt() - TWO63}, sumW: ${sumW}`);

        let sectionProof = undefined;

        for(let i = 0; i < SectionCount; i++) {
            let left = 0;
            let right = 0;
            let zLeft = Field(0);
            let zRight = Field(0);

            if(i != SectionCount - 1) {
                left = headId * headDim + i * SectionSize;
                right = headId * headDim + (i + 1) * SectionSize;
            } else {
                left = headId * headDim + (SectionCount - 1) * SectionSize;
                right = headId * headDim + headDim;
            }

            // console.log(`left: ${left}, right: ${right}`);

            zLeft = fastPow(z, rowId * n_local_heads * headDim + left);
            zRight = fastPow(z, rowId * n_local_heads * headDim + right);

            let input = new SectionInput({
                rowId: UInt32.from(rowId),

                left: UInt32.from(left),
                right: UInt32.from(right),
                zLeft,
                zRight,

                z,
                xmax,
                sum: sumW,
            })

            let xArr: UInt64[] = new Array(SectionSize);
            if(i != SectionCount - 1) {
                xArr = xx.slice(i * SectionSize, (i + 1) * SectionSize);
            } else {
                for(let k = (SectionCount - 1) * SectionSize; k < headDim; k++) {
                    xArr[k] = xx[k];
                }
                for(let k = headDim; k < SectionCount * SectionSize; k++) {
                    xArr[k] = UInt64.from(-(64n << 36n) + TWO63);
                }
            }

            // console.log('before Section.base(');

            const proof = await Section.base(input, xArr);
            // const proof = await Rope.base(input, xs, freqs, ys, rescale);
            // const proof = await NormX.base(input, xArr, wArr, qArr);
            const ok = await verify(proof.proof, vkSection);

            sectionProof = proof.proof;

            let proofResStr = JSON.stringify(proof.proof.toJSON());
            fs.mkdir(`proofs/pos_${posId}/layer_${layerId}/${name}_softmax/row_${rowId}`, { recursive: true }).then(() =>
                fs.writeFile(`proofs/pos_${posId}/layer_${layerId}/${name}_softmax/row_${rowId}/head_${n_local_heads - 1 + headId}_${i}.json`, proofResStr, "utf8")
            ).catch(console.error);

            console.log(`${nowPrefix()} ${name} Softmax Section base proof ${rowId} ${headId}_${i} verify result: ${ok}`);
        }

        let zStart = fastPow(z, rowId * n_local_heads * headDim + headId * headDim);
        let zEnd = fastPow(z, rowId * n_local_heads * headDim + (headId + 1) * headDim);

        let headInput = new HeadInput({
            rowId: UInt32.from(rowId),
            headDim: UInt32.from(headDim),

            headStart: UInt32.from(headId),
            headEnd: UInt32.from(headId + 1),

            zStart,
            zEnd,
            z,
        })

        if(sectionProof) {
            const proof = await Head.base(headInput, sectionProof);

            let proofResStr = JSON.stringify(proof.proof.toJSON());
            await fs.mkdir(`proofs/pos_${posId}/layer_${layerId}/${name}_softmax/row_${rowId}`, { recursive: true }).then(async () =>
                await fs.writeFile(`proofs/pos_${posId}/layer_${layerId}/${name}_softmax/row_${rowId}/head_base_${n_local_heads - 1 + headId}.json`, proofResStr, "utf8")
            ).catch(console.error);


            const start = performance.now();
            const ok = await verify(proof.proof, vkHead);
            const end = performance.now();

            console.log(`${nowPrefix()} ${name} Softmax Head base proof ${rowId} ${headId} verify result: ${ok}, verifying time: ${end - start} ms`);
        }
    }

    async function calcHeadMerge(name: string, posId: number, layerId: number, rowId: number, ind: number, headDim: number, vkHead: VerificationKey) {

        let proofs: HeadProofType[] = new Array(2 * n_local_heads - 1);
        for (let j = 0; j < n_local_heads; j++) {
            let proofFile = `proofs/pos_${posId}/layer_${layerId}/${name}_softmax/row_${rowId}/head_base_${n_local_heads - 1 + j}.json`;
            let proofStr = await readFile(proofFile);
            const proofJson = JSON.parse(proofStr) as HeadProofJSON;
            const proof = await HeadProof.fromJSON(proofJson);
            proofs[n_local_heads - 1 + j] = proof;
        }

        let count = 8;
        for(let j = ind; j >= 0; j--) {
            if(count == 0) break;
            count--;

            for(let k = j + 1; k < n_local_heads - 1; k++) {
                let proofFile = `proofs/pos_${posId}/layer_${layerId}/${name}_softmax/row_${rowId}/head_merge_${k}.json`;
                let proofStr = await readFile(proofFile);
                const proofJson = JSON.parse(proofStr) as HeadProofJSON;
                const proof = await HeadProof.fromJSON(proofJson);
                proofs[k] = proof;
            }

            let leftProof = proofs[2 * j + 1];
            let rightProof = proofs[2 * j + 2];
            let headInput = new HeadInput({
                rowId: UInt32.from(rowId),

                headDim: UInt32.from(headDim),

                headStart: leftProof.publicInput.headStart,
                headEnd: rightProof.publicInput.headEnd,

                zStart: leftProof.publicInput.zStart,
                zEnd: rightProof.publicInput.zEnd,

                z: leftProof.publicInput.z,
            });


            const proof = await Head.merge(headInput, leftProof, rightProof);

            const start = performance.now();
            const ok = await verify(proof.proof, vkHead);
            const end = performance.now();

            let proofResStr = JSON.stringify(proof.proof.toJSON());
            // console.log(proofResStr);

            try {
            // await fs.mkdir(`proofs/pos_${posId}/layer_${layerId}/${name}_softmax/row_${rowId}`, { recursive: true });
                await fs.writeFile(`proofs/pos_${posId}/layer_${layerId}/${name}_softmax/row_${rowId}/head_merge_${j}.json`, proofResStr, "utf8");
            } catch (error) {
                console.error("Error writing file:", error);
            }

            console.log(`${nowPrefix()} ${name} Softmax Head merge proof ${rowId} ${j} verify result: ${ok}, verifying time: ${end - start} ms`);

            // await sleep(5000);

        }
    }

    async function softmaxWrapRow(name: string, tokenListLen: number, posId: number, layerId: number, vkRows: VerificationKey) {
        for(let rowId = 0; rowId < tokenListLen; rowId++) {
            let proofFile = `proofs/pos_${posId}/layer_${layerId}/${name}_softmax/row_${rowId}/head_merge_0.json`;
            let proofStr = await readFile(proofFile);
            const proofJson = JSON.parse(proofStr) as RowsProofJSON;

            const earlierProof = await HeadProof.fromJSON(proofJson);
            // const ok = await verify(proof, vkNorm); 
            // assert(ok, `merge proof merge_${rowId}_${j} verify failed!`);

            let publicInput = earlierProof.publicInput;

            let rId = publicInput.rowId;

            let input = new RowsInput({
                rowStart: rId,
                rowEnd: rId.add(1),
                zRowStart: publicInput.zStart,
                zRowEnd: publicInput.zEnd,

                headDim: publicInput.headDim,
                headCount: UInt32.from(n_local_heads),

                z: publicInput.z,
            })

            const proof = await Rows.base(input, earlierProof);

            const start = performance.now();
            const ok = await verify(proof.proof, vkRows);
            const end = performance.now();

            console.log(`${nowPrefix()} Softmax ${name} Rows base proof verify: ${ok}, verifying time: ${end - start} ms`);

            let proofResStr = JSON.stringify(proof.proof.toJSON());
            await fs.mkdir(`proofs/pos_${posId}/layer_${layerId}/${name}_softmax/summary`, { recursive: true });
            await fs.writeFile(`proofs/pos_${posId}/layer_${layerId}/${name}_softmax/summary/wrap_row_${rowId}.json`, proofResStr, "utf8")

            console.log(`${nowPrefix()} Softmax ${name} row_${rowId} wrap proof verify result: ${ok}`);
        }
    }

    async function softmaxMergeRow(name: string, tokenListLen: number, posId: number, layerId: number, vkRows: VerificationKey) {
        let proofFile0 = `proofs/pos_${posId}/layer_${layerId}/${name}_softmax/summary/wrap_row_0.json`;
        let proofStr0 = await readFile(proofFile0);
        const proofJson0 = JSON.parse(proofStr0) as RowsProofJSON;
        let prevProof = await RowsProof.fromJSON(proofJson0);

        for (let i = 1; i < tokenListLen; i++) {
            let proofFile = `proofs/pos_${posId}/layer_${layerId}/${name}_softmax/summary/wrap_row_${i}.json`;
            let proofStr = await readFile(proofFile);
            const proofJson = JSON.parse(proofStr) as RowsProofJSON;
            const currProof = await RowsProof.fromJSON(proofJson);

            let input = new RowsInput({
                rowStart: prevProof.publicInput.rowStart,
                rowEnd: currProof.publicInput.rowEnd,
                zRowStart: prevProof.publicInput.zRowStart,
                zRowEnd: currProof.publicInput.zRowEnd,

                headDim: prevProof.publicInput.headDim,
                headCount: UInt32.from(n_local_heads),

                z: prevProof.publicInput.z,
            })

            const proof = await Rows.merge(input, prevProof, currProof);

            const start = performance.now();
            const ok = await verify(proof.proof, vkRows);
            const end = performance.now();

            console.log(`${nowPrefix()} Softmax ${name} Rows merge proof ${i} verify result: ${ok}, verifying time: ${end - start} ms`);
            prevProof = proof.proof;
        }

        const proof = prevProof;
        const ok = await verify(proof, vkRows);
        console.log(`${nowPrefix()} Softmax ${name} Rows merge proof verify result: ${ok}`);
    
        let proofStr = JSON.stringify(proof.toJSON())
        await fs.writeFile(`proofs/pos_${posId}/layer_${layerId}/${name}_softmax/summary/softmax.json`, proofStr, "utf8");
    }

    return {Head, compileSoftmaxHeadWithCache, calcHeadBase, calcHeadMerge,
            Rows, compileSoftmaxRowsWithCache, softmaxWrapRow, softmaxMergeRow}
}

// End Softmax ===========================================================================================

// Begin Sigmoid ===========================================================================================

function createSigmoidClass(name: string, Dim: number, ShortDim: number) {
    const LOG_TABLE_SIZE = 8n;

    const SectionUInt64 = Provable.Array(UInt64, ShortDim);
    const SectionInt64 = Provable.Array(Int64, ShortDim);
    const SectionCount = Dim / ShortDim;

    const LOG2E_Q23 = 12102203n;

    const TWO23 = 1n << 23n;
    const TWO46 = 1n << 46n;

    const q23Table = [
        UInt64.from(8388608n), UInt64.from(8411351n), UInt64.from(8434157n), UInt64.from(8457024n), UInt64.from(8479953n), UInt64.from(8502945n), UInt64.from(8525999n), UInt64.from(8549115n),
        UInt64.from(8572294n), UInt64.from(8595536n), UInt64.from(8618841n), UInt64.from(8642209n), UInt64.from(8665640n), UInt64.from(8689135n), UInt64.from(8712694n), UInt64.from(8736316n),
        UInt64.from(8760003n), UInt64.from(8783754n), UInt64.from(8807569n), UInt64.from(8831449n), UInt64.from(8855393n), UInt64.from(8879402n), UInt64.from(8903477n), UInt64.from(8927617n),
        UInt64.from(8951822n), UInt64.from(8976093n), UInt64.from(9000430n), UInt64.from(9024832n), UInt64.from(9049301n), UInt64.from(9073836n), UInt64.from(9098438n), UInt64.from(9123106n),
        UInt64.from(9147841n), UInt64.from(9172644n), UInt64.from(9197513n), UInt64.from(9222450n), UInt64.from(9247455n), UInt64.from(9272527n), UInt64.from(9297668n), UInt64.from(9322876n),
        UInt64.from(9348153n), UInt64.from(9373498n), UInt64.from(9398913n), UInt64.from(9424396n), UInt64.from(9449948n), UInt64.from(9475569n), UInt64.from(9501260n), UInt64.from(9527021n),
        UInt64.from(9552851n), UInt64.from(9578751n), UInt64.from(9604722n), UInt64.from(9630763n), UInt64.from(9656875n), UInt64.from(9683057n), UInt64.from(9709311n), UInt64.from(9735635n),
        UInt64.from(9762031n), UInt64.from(9788499n), UInt64.from(9815038n), UInt64.from(9841649n), UInt64.from(9868333n), UInt64.from(9895088n), UInt64.from(9921917n), UInt64.from(9948818n),

        UInt64.from(9975792n), UInt64.from(10002839n), UInt64.from(10029959n), UInt64.from(10057153n), UInt64.from(10084421n), UInt64.from(10111763n), UInt64.from(10139178n), UInt64.from(10166669n),
        UInt64.from(10194233n), UInt64.from(10221873n), UInt64.from(10249587n), UInt64.from(10277376n), UInt64.from(10305241n), UInt64.from(10333181n), UInt64.from(10361198n), UInt64.from(10389290n),
        UInt64.from(10417458n), UInt64.from(10445702n), UInt64.from(10474024n), UInt64.from(10502422n), UInt64.from(10530897n), UInt64.from(10559449n), UInt64.from(10588078n), UInt64.from(10616785n),
        UInt64.from(10645571n), UInt64.from(10674434n), UInt64.from(10703375n), UInt64.from(10732395n), UInt64.from(10761493n), UInt64.from(10790671n), UInt64.from(10819927n), UInt64.from(10849263n),
        UInt64.from(10878678n), UInt64.from(10908173n), UInt64.from(10937748n), UInt64.from(10967404n), UInt64.from(10997139n), UInt64.from(11026955n), UInt64.from(11056853n), UInt64.from(11086831n),
        UInt64.from(11116890n), UInt64.from(11147031n), UInt64.from(11177254n), UInt64.from(11207558n), UInt64.from(11237945n), UInt64.from(11268414n), UInt64.from(11298966n), UInt64.from(11329601n),
        UInt64.from(11360318n), UInt64.from(11391119n), UInt64.from(11422004n), UInt64.from(11452972n), UInt64.from(11484024n), UInt64.from(11515161n), UInt64.from(11546381n), UInt64.from(11577687n),
        UInt64.from(11609077n), UInt64.from(11640552n), UInt64.from(11672113n), UInt64.from(11703759n), UInt64.from(11735492n), UInt64.from(11767310n), UInt64.from(11799214n), UInt64.from(11831205n),

        UInt64.from(11863283n), UInt64.from(11895447n), UInt64.from(11927699n), UInt64.from(11960038n), UInt64.from(11992465n), UInt64.from(12024980n), UInt64.from(12057583n), UInt64.from(12090275n),
        UInt64.from(12123055n), UInt64.from(12155924n), UInt64.from(12188882n), UInt64.from(12221929n), UInt64.from(12255066n), UInt64.from(12288293n), UInt64.from(12321610n), UInt64.from(12355017n),
        UInt64.from(12388515n), UInt64.from(12422104n), UInt64.from(12455784n), UInt64.from(12489555n), UInt64.from(12523417n), UInt64.from(12557372n), UInt64.from(12591418n), UInt64.from(12625557n),
        UInt64.from(12659788n), UInt64.from(12694112n), UInt64.from(12728530n), UInt64.from(12763040n), UInt64.from(12797644n), UInt64.from(12832342n), UInt64.from(12867134n), UInt64.from(12902021n),
        UInt64.from(12937002n), UInt64.from(12972077n), UInt64.from(13007248n), UInt64.from(13042514n), UInt64.from(13077876n), UInt64.from(13113334n), UInt64.from(13148888n), UInt64.from(13184538n),
        UInt64.from(13220285n), UInt64.from(13256129n), UInt64.from(13292070n), UInt64.from(13328108n), UInt64.from(13364244n), UInt64.from(13400479n), UInt64.from(13436811n), UInt64.from(13473242n),
        UInt64.from(13509772n), UInt64.from(13546400n), UInt64.from(13583128n), UInt64.from(13619956n), UInt64.from(13656883n), UInt64.from(13693911n), UInt64.from(13731039n), UInt64.from(13768268n),
        UInt64.from(13805597n), UInt64.from(13843028n), UInt64.from(13880560n), UInt64.from(13918194n), UInt64.from(13955930n), UInt64.from(13993769n), UInt64.from(14031709n), UInt64.from(14069753n),

        UInt64.from(14107900n), UInt64.from(14146151n), UInt64.from(14184505n), UInt64.from(14222963n), UInt64.from(14261525n), UInt64.from(14300192n), UInt64.from(14338964n), UInt64.from(14377841n),
        UInt64.from(14416823n), UInt64.from(14455911n), UInt64.from(14495105n), UInt64.from(14534405n), UInt64.from(14573812n), UInt64.from(14613326n), UInt64.from(14652946n), UInt64.from(14692675n),
        UInt64.from(14732510n), UInt64.from(14772454n), UInt64.from(14812506n), UInt64.from(14852667n), UInt64.from(14892937n), UInt64.from(14933316n), UInt64.from(14973804n), UInt64.from(15014402n),
        UInt64.from(15055110n), UInt64.from(15095929n), UInt64.from(15136858n), UInt64.from(15177898n), UInt64.from(15219050n), UInt64.from(15260313n), UInt64.from(15301688n), UInt64.from(15343175n),
        UInt64.from(15384774n), UInt64.from(15426487n), UInt64.from(15468312n), UInt64.from(15510251n), UInt64.from(15552304n), UInt64.from(15594470n), UInt64.from(15636751n), UInt64.from(15679147n),
        UInt64.from(15721657n), UInt64.from(15764283n), UInt64.from(15807024n), UInt64.from(15849881n), UInt64.from(15892855n), UInt64.from(15935945n), UInt64.from(15979151n), UInt64.from(16022475n),
        UInt64.from(16065917n), UInt64.from(16109476n), UInt64.from(16153153n), UInt64.from(16196949n), UInt64.from(16240863n), UInt64.from(16284896n), UInt64.from(16329049n), UInt64.from(16373322n),
        UInt64.from(16417714n), UInt64.from(16462227n), UInt64.from(16506861n), UInt64.from(16551616n), UInt64.from(16596492n), UInt64.from(16641489n), UInt64.from(16686609n), UInt64.from(16731851n),
    ]

    function pickByIndex(idx: UInt64): UInt64 {
        const mask = q23Table.map((_, i) => idx.equals(UInt64.from(i))); // Bool[]
        return Provable.switch(mask, UInt64, q23Table); // 只允许一个 true
    }

    function shrVar(x: UInt64, k: UInt64): UInt64 {
        // 取 k 的低 6 位作为移位量（0..63）
        const kbits = k.toBits(); // Bool[32]，我们只用前 6 位
        const b0 = kbits[0], b1 = kbits[1], b2 = kbits[2], b3 = kbits[3], b4 = kbits[4], b5 = kbits[5];

        // 逐级条件移位：1,2,4,8,16,32
        let v = x;
        const v1  = v.rightShift(1);  v = Provable.if(b0, v1, v);
        const v2  = v.rightShift(2);  v = Provable.if(b1, v2, v);
        const v4  = v.rightShift(4);  v = Provable.if(b2, v4, v);
        const v8  = v.rightShift(8);  v = Provable.if(b3, v8, v);
        const v16 = v.rightShift(16);  v = Provable.if(b4, v16, v);
        const v32 = v.rightShift(32);  v = Provable.if(b5, v32, v);

        return v; // 相当于 x >>> k（逻辑右移）
    }

    function shlVar(x: UInt64, k: UInt64): UInt64 {
        // 取 k 的低 6 位作为移位量（0..63）
        const kbits = k.toBits(); // Bool[32]，我们只用前 6 位
        const b0 = kbits[0], b1 = kbits[1], b2 = kbits[2], b3 = kbits[3], b4 = kbits[4], b5 = kbits[5];

        // 逐级条件移位：1,2,4,8,16,32
        let v = x;
        const v1  = v.leftShift(1);  v = Provable.if(b0, v1, v);
        const v2  = v.leftShift(2);  v = Provable.if(b1, v2, v);
        const v4  = v.leftShift(4);  v = Provable.if(b2, v4, v);
        const v8  = v.leftShift(8);  v = Provable.if(b3, v8, v);
        const v16 = v.leftShift(16);  v = Provable.if(b4, v16, v);
        const v32 = v.leftShift(32);  v = Provable.if(b5, v32, v);

        return v; // 相当于 x <<< k（逻辑左移）
    }

    class SectionInput extends Struct({
        rowId: UInt32,

        left: UInt32,
        right: UInt32,
        zLeft: Field,
        zRight: Field,

        z: Field,
    }) {}

    class SectionOutput extends Struct({
        zsumX: Field,
        zsumY: Field,
    }) {}

    class DATA extends Struct({ q: Int64, r: UInt64, k: Int64, f: UInt64, f2: UInt64, r2: UInt64 }) {}

    const Section = ZkProgram({
        name: 'Section',
        publicInput: SectionInput,
        publicOutput: SectionOutput,
        methods: {
            base_sigmoid: {
                privateInputs: [UInt64, SectionInt64, SectionUInt64],
                async method(input: SectionInput, rescale: UInt64, xs: Int64[], rs: UInt64[]) {
                    input.right.assertEquals(input.left.add(ShortDim));

                    let zsumX = Field(0);
                    let zsumY = Field(0);

                    let zi = input.zLeft;
                    for(let i = 0; i < ShortDim; i++) {
                        let x = xs[i];
                        let r = rs[i];

                        let xField = x.toField();
                        r.assertLessThan(rescale);
                        let lastOutput = rescale.value.mul(xField).add(r.value);
                        zsumX = zsumX.add(zi.mul(lastOutput));

                        // Provable.asProver(() => {
                        //     console.log(`${input.rowId}, ${input.left.add(i)}: x: ${x}, x0: ${lastOutput}`);
                        // });

                        let product = x.neg().mul(LOG2E_Q23);

                        let pdt = 0n;
                        Provable.asProver(() => {
                            pdt = product.toBigint();
                        });

                        const qr = Provable.witness(DATA, () => {
                            let q = pdt >> 23n;
                            let r1 = pdt & (TWO23 - 1n);
                            let k = q >> 23n;
                            let f = q & (TWO23 - 1n);
                            let f2 = f >> (23n - LOG_TABLE_SIZE);
                            let r2 = f & ((1n << (23n - LOG_TABLE_SIZE)) - 1n);
                            let t = pickByIndex(UInt64.from(f2));

                            let u = 0n;
                            let ur = 0n;
                            if(k < 0) {
                                u = t.toBigInt() >> (-k);
                                ur = t.toBigInt() & ((1n << (-k)) - 1n);
                            } else {
                                u = t.toBigInt() << k;
                            }
                            // console.log(`DATA: x: ${x}, pdt=${pdt}, y=${q}, k: ${k}, f: ${f}, t: ${t}, u: ${u}`);
                            return new DATA({ q: Int64.from(q), r: UInt64.from(r1), k: Int64.from(k), f: UInt64.from(f),
                                        f2: UInt64.from(f2), r2: UInt64.from(r2) });
                        });

                        let y = qr.q;
                        let r1 = qr.r;
                        product.toField().assertEquals(y.toField().mul(TWO23).add(r1.value));
                        r1.assertLessThan(UInt64.from(TWO23));

                        let k = qr.k;
                        let f = qr.f;
                        y.toField().assertEquals(k.toField().mul(TWO23).add(f.value));
                        f.assertLessThan(UInt64.from(TWO23));

                        let f2 = qr.f2;
                        let r2 = qr.r2;
                        let b2 = UInt64.from(1n << (23n - LOG_TABLE_SIZE));
                        f.value.assertEquals(f2.value.mul(b2.value).add(r2.value));
                        r2.assertLessThan(b2);

                        let t = pickByIndex(f2);

                        let u = Provable.if(k.isNegative(), shrVar(t, k.magnitude), shlVar(t, k.magnitude));
                        // let u = UInt64.from(0);
                        u = Provable.if(k.isNegative() && k.magnitude.greaterThanOrEqual(UInt64.from(63)), UInt64.from(0), u);

                        let q = UInt64.from(TWO46).div(UInt64.from(TWO23).add(u));

                        // let c = x.mul(q).div(TWO23);

                        // Provable.asProver(() => {
                        //     console.log(`u: ${u}, q: ${q}`);
                        // });

                        let qzi = zi.mul(q.value);
                        zsumY = zsumY.add(qzi);

                        zi = zi.mul(input.z);
                    };
                    zi.assertEquals(input.zRight, 'zLeft zRight not match');

                    const out = new RowsOutput({
                        zsumX,
                        zsumY,
                      });
                    return {publicOutput: out};
                }
            },
            base_silu: {
                privateInputs: [UInt64, SectionInt64, SectionUInt64],
                async method(input: SectionInput, rescale: UInt64, xs: Int64[], rs: UInt64[]) {
                    input.right.assertEquals(input.left.add(ShortDim));

                    let zsumX = Field(0);
                    let zsumY = Field(0);

                    let zi = input.zLeft;
                    for(let i = 0; i < ShortDim; i++) {
                        let x = xs[i];
                        let r = rs[i];

                        let xField = x.toField();
                        r.assertLessThan(rescale);
                        let lastOutput = rescale.value.mul(xField).add(r.value);
                        zsumX = zsumX.add(zi.mul(lastOutput));

                        // Provable.asProver(() => {
                        //     console.log(`${input.rowId}, ${input.left.add(i)}: x: ${x}, x0: ${lastOutput}`);
                        // });

                        let product = x.neg().mul(LOG2E_Q23);

                        let pdt = 0n;
                        Provable.asProver(() => {
                            pdt = product.toBigint();
                        });

                        const qr = Provable.witness(DATA, () => {
                            let q = pdt >> 23n;
                            let r1 = pdt & (TWO23 - 1n);
                            let k = q >> 23n;
                            let f = q & (TWO23 - 1n);
                            let f2 = f >> (23n - LOG_TABLE_SIZE);
                            let r2 = f & ((1n << (23n - LOG_TABLE_SIZE)) - 1n);
                            let t = pickByIndex(UInt64.from(f2));

                            let u = 0n;
                            let ur = 0n;
                            if(k < 0) {
                                u = t.toBigInt() >> (-k);
                                ur = t.toBigInt() & ((1n << (-k)) - 1n);
                            } else {
                                u = t.toBigInt() << k;
                            }
                            // console.log(`DATA: x: ${x}, pdt=${pdt}, y=${q}, k: ${k}, f: ${f}, t: ${t}, u: ${u}`);
                            return new DATA({ q: Int64.from(q), r: UInt64.from(r1), k: Int64.from(k), f: UInt64.from(f),
                                        f2: UInt64.from(f2), r2: UInt64.from(r2) });
                        });

                        let y = qr.q;
                        let r1 = qr.r;
                        product.toField().assertEquals(y.toField().mul(TWO23).add(r1.value));
                        r1.assertLessThan(UInt64.from(TWO23));

                        let k = qr.k;
                        let f = qr.f;
                        y.toField().assertEquals(k.toField().mul(TWO23).add(f.value));
                        f.assertLessThan(UInt64.from(TWO23));

                        let f2 = qr.f2;
                        let r2 = qr.r2;
                        let b2 = UInt64.from(1n << (23n - LOG_TABLE_SIZE));
                        f.value.assertEquals(f2.value.mul(b2.value).add(r2.value));
                        r2.assertLessThan(b2);

                        let t = pickByIndex(f2);

                        let u = Provable.if(k.isNegative(), shrVar(t, k.magnitude), shlVar(t, k.magnitude));
                        // let u = UInt64.from(0);
                        u = Provable.if(k.isNegative() && k.magnitude.greaterThanOrEqual(UInt64.from(63)), UInt64.from(0), u);

                        let q = UInt64.from(TWO46).div(UInt64.from(TWO23).add(u));

                        let c = x.mul(q).div(TWO23);

                        // Provable.asProver(() => {
                        //     console.log(`u: ${u}, q: ${q}`);
                        // });

                        let czi = zi.mul(c.toField());
                        zsumY = zsumY.add(czi);

                        zi = zi.mul(input.z);
                    };
                    zi.assertEquals(input.zRight, 'zLeft zRight not match');

                    const out = new RowsOutput({
                        zsumX,
                        zsumY,
                      });
                    return {publicOutput: out};
                }
            },
            merge: {
                privateInputs: [SelfProof, SelfProof],
                async method(input: SectionInput,
                    leftProof: InstanceType<typeof SelfProof<SectionInput, SectionOutput> > ,
                    rightProof: InstanceType<typeof SelfProof<SectionInput, SectionOutput> >) {

                    leftProof.verify();
                    rightProof.verify();

                    let leftInput = leftProof.publicInput;
                    let leftOutput = leftProof.publicOutput;
                    let rightInput = rightProof.publicInput;
                    let rightOutput = rightProof.publicOutput;

                    leftInput.rowId.assertEquals(rightInput.rowId);

                    input.left.assertEquals(leftInput.left);
                    input.right.assertEquals(rightInput.right);
                    leftInput.right.assertEquals(rightInput.left);

                    input.zLeft.assertEquals(leftInput.zLeft, 'zLeft not equal');
                    input.zRight.assertEquals(rightInput.zRight, 'zRight not equal');
                    leftInput.zRight.assertEquals(rightInput.zLeft, 'zLeft zRight not equal');

                    input.z.assertEquals(leftInput.z, 'z 1 not equal');
                    input.z.assertEquals(rightInput.z, 'z 2 not equal');

                    let zsumX = leftOutput.zsumX.add(rightOutput.zsumX);
                    let zsumY = leftOutput.zsumY.add(rightOutput.zsumY);

                    const out = new RowsOutput({
                        zsumX,
                        zsumY,
                        });
                    return {publicOutput: out};
                }
            }
        }
    })

    const SectionProof = ZkProgram.Proof(Section);
    type SectionProofType = InstanceType<typeof SectionProof>;
    type SectionProofJSON = ReturnType<InstanceType<typeof SectionProof>["toJSON"]>;


    async function compileSigmoidSectionWithCache() {
        const cache = Cache.FileSystem(`./o1js-cache/Sigmoid-section-${name}`);

        // let sectionStr = await Section.analyzeMethods();
        // console.log('Sigmoid Section info: ', sectionStr);
    
        return Section.compile({cache: cache});
    }

    async function calcSectionBase(name: string, posId: number, layerId: number, rowId: number, vkSection: VerificationKey) {
        let xs: Int64[][] = [];
        const bufX = await readBinary(`${zkDataDir}/pos_${posId}/layer_${layerId}/sigmoid_${name}_x.bin`);
        const xData = bufferToInt64ArrayLE(bufX);
        for(let i = 0; i < xData.length; i += Dim) {
            xs.push(xData.slice(i, i + Dim));
        }

        let rs: Int64[][] = [];
        const bufR = await readBinary(`${zkDataDir}/pos_${posId}/layer_${layerId}/sigmoid_${name}_r.bin`);
        const rData = bufferToInt64ArrayLE(bufR);
        for(let i = 0; i < rData.length; i += Dim) {
            rs.push(rData.slice(i, i + Dim));
        }
        let rrs = rs.map(rs => rs.map(r => UInt64.from(r.toBigint())));

        let zStr: string = await readFile('proofs/embed/hash');
        let z = Field(zStr);

        let zd = fastPow(z, ShortDim);

        for(let i = 0; i < SectionCount; i++) {
            let left = i * ShortDim;
            let right = (i + 1) * ShortDim;

            let input = new SectionInput({
                rowId: UInt32.from(rowId),
                left: UInt32.from(left),
                right: UInt32.from(right),
                zLeft: fastPow(z, rowId * Dim + left),
                zRight: fastPow(z, rowId * Dim + right),
                z,
            })

            const proof = await Section.base_sigmoid(input, UInt64.from(1n << 33n), xs[rowId].slice(left, right), rrs[rowId].slice(left, right));
            // const proof = await Rope.base(input, xs, freqs, ys, rescale);
            // const proof = await NormX.base(input, xArr, wArr, qArr);

            const start = performance.now();
            const ok = await verify(proof.proof, vkSection);
            const end = performance.now();

            let proofResStr = JSON.stringify(proof.proof.toJSON());
            fs.mkdir(`proofs/pos_${posId}/layer_${layerId}/sigmoid_${name}/row_${rowId}`, { recursive: true }).then(() =>
                fs.writeFile(`proofs/pos_${posId}/layer_${layerId}/sigmoid_${name}/row_${rowId}/section_${SectionCount - 1 + i}.json`, proofResStr, "utf8")
            ).catch(console.error);

            console.log(`${nowPrefix()} Sigmoid_${name} Section base proof ${rowId}_${SectionCount - 1 + i} verify result: ${ok}, verifying time: ${end - start} ms`);
        }
    }


    async function calcSectionMerge(name: string, posId: number, layerId: number, rowId: number, vkSection: VerificationKey) {
        let proofs: SectionProofType[] = new Array(2 * SectionCount - 1);
        for (let j = 0; j < SectionCount; j++) {
            let proofFile = `proofs/pos_${posId}/layer_${layerId}/sigmoid_${name}/row_${rowId}/section_${SectionCount - 1 + j}.json`;
            let proofStr = await readFile(proofFile);
            const proofJson = JSON.parse(proofStr) as SectionProofJSON;
            const proof = await SectionProof.fromJSON(proofJson);
            proofs[SectionCount - 1 + j] = proof;
        }

        for(let j = SectionCount - 2; j >= 0; j--) {
            let leftProof = proofs[2 * j + 1];
            let rightProof = proofs[2 * j + 2];
            let input = new SectionInput({
                rowId: leftProof.publicInput.rowId,
                left: leftProof.publicInput.left,
                right: rightProof.publicInput.right,
                zLeft: leftProof.publicInput.zLeft,
                zRight: rightProof.publicInput.zRight,
                z: leftProof.publicInput.z,
            });

            const proof = await Section.merge(input, leftProof, rightProof);

            const start = performance.now();
            const ok = await verify(proof.proof, vkSection);
            const end = performance.now();

            console.log(`${nowPrefix()} Sigmoid_${name} Section merge proof ${rowId} ${j} verify result: ${ok}, verifying time: ${end - start} ms`);
            proofs[j] = proof.proof;
        }

        let proofStr = JSON.stringify(proofs[0].toJSON())
        await fs.writeFile(`proofs/pos_${posId}/layer_${layerId}/sigmoid_${name}/row_${rowId}/section_merge_0.json`, proofStr, "utf8");
    }

    class RowsInput extends Struct({
        rowStart: UInt32,
        rowEnd: UInt32,
        zRowStart: Field,
        zRowEnd: Field,

        z: Field,
    }) {}

    class RowsOutput extends Struct({
        zsumX: Field,
        zsumY: Field,
    }) {}

    const Rows = ZkProgram({
        name: 'Rows',
        publicInput: RowsInput,
        publicOutput: RowsOutput,
        methods: {
            base: {
                privateInputs: [SectionProof],
                async method(input: RowsInput, proof: SectionProofType) {
                    proof.verify();

                    let pInput = proof.publicInput;
                    let pOutput = proof.publicOutput;

                    input.rowStart.assertEquals(pInput.rowId);
                    input.rowEnd.assertEquals(input.rowStart.add(1));

                    input.zRowStart.assertEquals(pInput.zLeft);
                    input.zRowEnd.assertEquals(pInput.zRight);

                    input.z.assertEquals(pInput.z, 'z not equal');

                    const out = new RowsOutput({
                        zsumX: pOutput.zsumX,
                        zsumY: pOutput.zsumY,
                      });
                    return {publicOutput: out};
                }
            },
            merge: {
                privateInputs: [SelfProof, SelfProof],
                async method(input: RowsInput,
                    upProof: InstanceType<typeof SelfProof<RowsInput, RowsOutput> > ,
                    downProof: InstanceType<typeof SelfProof<RowsInput, RowsOutput> >) {

                    upProof.verify();
                    downProof.verify();

                    let upInput = upProof.publicInput;
                    let upOutput = upProof.publicOutput;
                    let downInput = downProof.publicInput;
                    let downOutput = downProof.publicOutput;

                    input.rowStart.assertEquals(upInput.rowStart);
                    input.rowEnd.assertEquals(downInput.rowEnd);
                    upInput.rowEnd.assertEquals(downInput.rowStart);

                    input.zRowStart.assertEquals(upInput.zRowStart, 'zRowStart not equal');
                    input.zRowEnd.assertEquals(downInput.zRowEnd, 'zRowEnd not equal');
                    upInput.zRowEnd.assertEquals(downInput.zRowStart, 'zRowStart zRowEnd not equal');

                    input.z.assertEquals(upInput.z, 'z 1 not equal');
                    input.z.assertEquals(downInput.z, 'z 2 not equal');

                    let zsumX = upOutput.zsumX.add(downOutput.zsumX);
                    let zsumY = upOutput.zsumY.add(downOutput.zsumY);

                    const out = new RowsOutput({
                        zsumX,
                        zsumY,
                        });
                    return {publicOutput: out};
                }
            }
        }
    })

    const RowsProof = ZkProgram.Proof(Rows);
    type RowsProofType = InstanceType<typeof RowsProof>;
    type RowsProofJSON = ReturnType<InstanceType<typeof RowsProof>["toJSON"]>;


    async function compileSigmoidRowsWithCache() {
        const cache = Cache.FileSystem(`./o1js-cache/Sigmoid-rows-${name}`);

        await compileSigmoidSectionWithCache();

        // let rowsStr = await Rows.analyzeMethods();
        // console.log(`${nowPrefix()} Sigmoid Rows info: `, rowsStr);

        return Rows.compile({cache: cache});
    }

    async function calcRowsBase(name: string, rowsLen: number, posId: number, layerId: number, vkRows: VerificationKey) {
        let zStr: string = await readFile('proofs/embed/hash');
        let z = Field(zStr);

        let zd = fastPow(z, Dim);

        for(let i = 0; i < rowsLen; i++) {
            let input = new RowsInput({
                rowStart: UInt32.from(i),
                rowEnd: UInt32.from(i + 1),
                zRowStart: fastPow(zd, i),
                zRowEnd: fastPow(zd, i + 1),
                z,
            })

            let proofFile = `proofs/pos_${posId}/layer_${layerId}/sigmoid_${name}/row_${i}/section_merge_0.json`;
            let proofStr = await readFile(proofFile);
            const proofJson = JSON.parse(proofStr) as SectionProofJSON;
            const earlierProof = await SectionProof.fromJSON(proofJson);

            const proof = await Rows.base(input, earlierProof);

            const start = performance.now();
            const ok = await verify(proof.proof, vkRows);
            const end = performance.now();

            let proofResStr = JSON.stringify(proof.proof.toJSON());
            fs.mkdir(`proofs/pos_${posId}/layer_${layerId}/sigmoid_${name}/summary`, { recursive: true }).then(() =>
                fs.writeFile(`proofs/pos_${posId}/layer_${layerId}/sigmoid_${name}/summary/row_${i}.json`, proofResStr, "utf8")
            ).catch(console.error);

            console.log(`${nowPrefix()} Sigmoid_${name} Rows base proof ${i} verify result: ${ok}, verifying time: ${end - start} ms`);
        }
    }


    async function calcRowMerge(name: string, rowsLen: number, posId: number, layerId: number, vkRows: VerificationKey) {
        let proofFile0 = `proofs/pos_${posId}/layer_${layerId}/sigmoid_${name}/summary/row_0.json`;
        let proofStr0 = await readFile(proofFile0);
        const proofJson0 = JSON.parse(proofStr0) as RowsProofJSON;
        let prevProof = await RowsProof.fromJSON(proofJson0);

        for (let i = 1; i < rowsLen; i++) {
            let proofFile = `proofs/pos_${posId}/layer_${layerId}/sigmoid_${name}/summary/row_${i}.json`;
            let proofStr = await readFile(proofFile);
            const proofJson = JSON.parse(proofStr) as RowsProofJSON;
            const currProof = await RowsProof.fromJSON(proofJson);

            let rowsInput = new RowsInput({
                rowStart: prevProof.publicInput.rowStart,
                rowEnd: currProof.publicInput.rowEnd,
                zRowStart: prevProof.publicInput.zRowStart,
                zRowEnd: currProof.publicInput.zRowEnd,
                z: prevProof.publicInput.z,
            })

            const proof = await Rows.merge(rowsInput, prevProof, currProof);

            const start = performance.now();
            const ok = await verify(proof.proof, vkRows);
            const end = performance.now();

            console.log(`${nowPrefix()} Sigmoid_${name} Rows merge proof ${i} verify result: ${ok}, verifying time: ${end - start} ms`);
            prevProof = proof.proof;
        }

        const proof = prevProof;
        const ok = await verify(proof, vkRows);
        console.log(`${nowPrefix()} Sigmoid_${name} Rows merge proof verify result: ${ok}`);
    
        let proofStr = JSON.stringify(proof.toJSON())
        await fs.writeFile(`proofs/pos_${posId}/layer_${layerId}/sigmoid_${name}/summary/sigmoid_${name}.json`, proofStr, "utf8");
    }

    // return {Rows, compileSigmoidWithCache, calcRowsBase, calcRowMerge}
    return {Section, compileSigmoidSectionWithCache, calcSectionBase, calcSectionMerge,
            Rows, compileSigmoidRowsWithCache, calcRowsBase, calcRowMerge }
}

// End Sigmoid ===========================================================================================


// Begin ExpertsSelector ===========================================================================================
function createExpertsSelectorClass(name: string, GroupDim: number, GroupCount: number) {
    class GroupInput extends Struct({
        rowId: UInt32,

        groupIdStart: UInt32,
        groupIdEnd: UInt32,

        zLeft: Field,
        zRight: Field,

        z: Field,
    }) {}

    class IdxValue extends Struct({
        idx: UInt32,
        val0: Int64,
        bias: Int64,
        val: UInt64,
    }) {}

    class GroupOutput extends Struct({
        largest2: UInt64,
        // largest8: Provable.Array(IdxValue, 8),
        cxs2: Field,
        cxs8: Field,
        cxs32: Field,
        zsumX: Field,
        hashBias: Field,
    }) {}

    const GroupUInt64 = Provable.Array(IdxValue, GroupDim);

    const Group = ZkProgram({
        name: 'Group',
        publicInput: GroupInput,
        publicOutput: GroupOutput,
        methods: {
            base: {
                privateInputs: [GroupUInt64, GroupUInt64],
                async method(input: GroupInput, xs: IdxValue[], xsSorted: IdxValue[]) {
                    input.groupIdEnd.assertEquals(input.groupIdStart.add(1));
                    let left = input.groupIdStart.mul(GroupDim);

                    let z = input.z;
                    let c = z.neg();

                    let x0 = xs[0];
                    let xSorted0 = xsSorted[0];
                    x0.val0.add(x0.bias).assertEquals(x0.val, 'original_scores + bias != scores');

                    let h1 = Poseidon.hash([x0.idx.value, x0.val0.toField(), x0.val.value]);
                    let cxs = h1.add(c);

                    let h2 = Poseidon.hash([xSorted0.idx.value, xSorted0.val0.toField(), xSorted0.val.value]);
                    let cxsSorted = h2.add(c);
                    // Provable.asProver(() => {
                    //     console.log(`xSorted0: ${xSorted0.idx.toBigint()}, ${xSorted0.val0.toBigint()}, ${xSorted0.val.toBigInt()}, ${h2.toBigInt()}`);
                    // });

                    let zi = input.zLeft;
                    let zsumX = zi.mul(x0.val0.toField());
                    zi = zi.mul(z);

                    x0.idx.assertEquals(left);

                    for(let i = 1; i < 8; i++) {
                        let x = xs[i];
                        let xSorted = xsSorted[i];

                        x.val0.add(x.bias).assertEquals(x.val, 'original_scores + bias != scores');

                        h1 = Poseidon.hash([x.idx.value, x.val0.toField(), x.val.value]);
                        cxs = cxs.mul(h1.add(c));
                        h2 = Poseidon.hash([xSorted.idx.value, xSorted.val0.toField(), xSorted.val.value]);
                        cxsSorted = cxsSorted.mul(h2.add(c));
                        // Provable.asProver(() => {
                        //     console.log(`xSorted[${i}]: ${h2.toBigInt()}`);
                        // });

                        x.idx.assertEquals(left.add(i));
                        xsSorted[i-1].val.assertGreaterThanOrEqual(xSorted.val);

                        zsumX = zsumX.add(zi.mul(x.val0.toField()));

                        zi = zi.mul(z);
                    };
                    let cxs8 = cxsSorted;

                    for(let i = 8; i < GroupDim; i++) {
                        let x = xs[i];
                        let xSorted = xsSorted[i];

                        x.val0.add(x.bias).assertEquals(x.val, 'original_scores + bias != scores');

                        cxs = cxs.mul(Poseidon.hash([x.idx.value, x.val0.toField(), x.val.value]).add(c))
                        cxsSorted = cxsSorted.mul(Poseidon.hash([xSorted.idx.value, xSorted.val0.toField(), xSorted.val.value]).add(c));

                        x.idx.assertEquals(left.add(i));
                        xsSorted[i-1].val.assertGreaterThanOrEqual(xSorted.val);

                        zsumX = zsumX.add(zi.mul(x.val0.toField()));

                        zi = zi.mul(input.z);
                    };

                    zi.assertEquals(input.zRight, 'zLeft zRight not match');
                    cxs.assertEquals(cxsSorted, 'cxs cxsSorted not match');

                    let sum2 = xsSorted[0].val.add(xsSorted[1].val);

                    let biases = new Array<Field>(GroupDim);
                    for(let i = 0; i < GroupDim; i++) {
                        biases[i] = xs[i].bias.toField();
                    }

                    let hashBias = Poseidon.hash(biases);

                    const out = new GroupOutput({
                        largest2: sum2,
                        cxs2: c.add(sum2.value), // For checking the sequence of the sum of the first 2 items
                        cxs8,
                        cxs32: cxs,
                        zsumX,
                        hashBias,
                      });
                    return {publicOutput: out};
                }
            },
            merge: {
                privateInputs: [SelfProof, SelfProof],
                async method(input: GroupInput,
                    leftProof: InstanceType<typeof SelfProof<GroupInput, GroupOutput> > ,
                    rightProof: InstanceType<typeof SelfProof<GroupInput, GroupOutput> >) {

                    leftProof.verify();
                    rightProof.verify();

                    let leftInput = leftProof.publicInput;
                    let leftOutput = leftProof.publicOutput;
                    let rightInput = rightProof.publicInput;
                    let rightOutput = rightProof.publicOutput;

                    leftInput.rowId.assertEquals(rightInput.rowId);

                    input.groupIdStart.assertEquals(leftInput.groupIdStart);
                    input.groupIdEnd.assertEquals(rightInput.groupIdEnd);
                    leftInput.groupIdEnd.assertEquals(rightInput.groupIdStart);

                    input.zLeft.assertEquals(leftInput.zLeft, 'zLeft not equal');
                    input.zRight.assertEquals(rightInput.zRight, 'zRight not equal');
                    leftInput.zRight.assertEquals(rightInput.zLeft, 'zLeft zRight not equal');

                    input.z.assertEquals(leftInput.z, 'z 1 not equal');
                    input.z.assertEquals(rightInput.z, 'z 2 not equal');


                    let cxs2 = leftOutput.cxs2.mul(rightOutput.cxs2);
                    let cxs8 = leftOutput.cxs8.mul(rightOutput.cxs8);
                    let cxs32 = leftOutput.cxs32.mul(rightOutput.cxs32);
                    let zsumX = leftOutput.zsumX.add(rightOutput.zsumX);

                    const out = new GroupOutput({
                        largest2: UInt64.from(0),
                        cxs2, cxs8, cxs32, zsumX,
                        hashBias: Poseidon.hash([leftOutput.hashBias, rightOutput.hashBias]),
                      });
                    return {publicOutput: out};
                }
            }
        }
    })

    const GroupProof = ZkProgram.Proof(Group);
    type GroupProofType = InstanceType<typeof GroupProof>;
    type GroupProofJSON = ReturnType<InstanceType<typeof GroupProof>["toJSON"]>;


    async function compileGroupWithCache() {
        const cache = Cache.FileSystem(`./o1js-cache/Gate-Group`);

        // let str = await Group.analyzeMethods();
        // console.log('Gate Group info: ', str);

        return Group.compile({cache: cache});
    }

    async function calcGroupBase(name: string, posId: number, layerId: number, rowId: number, vkGroup: VerificationKey) {
        let Dim = GroupDim * GroupCount;

        let xs: Int64[][] = [];
        const bufX = await readBinary(`${zkDataDir}/pos_${posId}/layer_${layerId}/gate_original_scores.bin`);
        const xData = bufferToInt64ArrayLE(bufX);
        for(let i = 0; i < xData.length; i += Dim) {
            xs.push(xData.slice(i, i + Dim));
        }

        const bufB = await readBinary(`${zkDataDir}/pos_${posId}/layer_${layerId}/gate_bias.bin`);
        const bs0 = bufferToUInt32ArrayLE(bufB);
        const bs = bs0.map(x => asInt64(x));

        let items: IdxValue[][] = [];
        for(let i = 0; i < xs.length; i++) {
            let row: IdxValue[] = [];
            for(let j = 0; j < Dim; j++) {
                let x = xs[i][j];
                let b = bs[j];
                // console.log(`x: ${x}, b:${b}`);
                let v = x.add(b);
                if(v < Int64.from(0)) {
                    console.log(`!!! scores[${i}][${j}] = ${v} < 0 !!!`);
                }
                let vv = UInt64.from(v.toBigint());
                let item = new IdxValue({
                    idx: UInt32.from(j),
                    val0: Int64.from(x),
                    bias: Int64.from(b),
                    val: vv,
                });
                row.push(item);
            }
            items.push(row);
        }

        let zStr: string = await readFile('proofs/embed/hash');
        let z = Field(zStr);

        // let zd = fastPow(z, Dim);

        for(let i = 0; i < GroupCount; i++) {
            let left = i * GroupDim;
            let right = (i+1) * GroupDim;

            let input = new GroupInput({
                rowId: UInt32.from(rowId),
                groupIdStart: UInt32.from(i),
                groupIdEnd: UInt32.from(i+1),
                zLeft: fastPow(z, rowId * Dim + left),
                zRight: fastPow(z, rowId * Dim + right),
                z,
            })

            let xItems = items[rowId].slice(left, right);
            const sortedItems = xItems.slice().sort((a, b) => {
                const vala = a.val.toBigInt();
                const valb = b.val.toBigInt();
                if (vala < valb)
                    return 1;
                else if (vala > valb)
                    return -1;
                return 0;
            });

            // console.log('xItems: ', xItems.map(x => x.val.toBigInt()));
            // console.log('sortedItems: ', sortedItems.map(x => x.val.toBigInt()));

            const proof = await Group.base(input, xItems, sortedItems);
            // const proof = await Rope.base(input, xs, freqs, ys, rescale);
            // const proof = await NormX.base(input, xArr, wArr, qArr);

            const start = performance.now();
            const ok = await verify(proof.proof, vkGroup);
            const end = performance.now();

            let proofResStr = JSON.stringify(proof.proof.toJSON());
            fs.mkdir(`proofs/pos_${posId}/layer_${layerId}/gate_group/row_${rowId}`, { recursive: true }).then(() =>
                fs.writeFile(`proofs/pos_${posId}/layer_${layerId}/gate_group/row_${rowId}/group_base_${GroupCount - 1 + i}.json`, proofResStr, "utf8")
            ).catch(console.error);

            console.log(`${nowPrefix()} Gate Group base proof ${rowId}_${GroupCount - 1 + i} verify result: ${ok}, verifying time: ${end - start} ms`);
        }
    }


    async function calcGroupMerge(name: string, posId: number, layerId: number, rowId: number, vkGroup: VerificationKey) {
        let proofs: GroupProofType[] = new Array(2 * GroupCount - 1);
        for (let j = 0; j < GroupCount; j++) {
            let proofFile = `proofs/pos_${posId}/layer_${layerId}/gate_group/row_${rowId}/group_base_${GroupCount - 1 + j}.json`;
            let proofStr = await readFile(proofFile);
            const proofJson = JSON.parse(proofStr) as GroupProofJSON;
            const proof = await GroupProof.fromJSON(proofJson);
            proofs[GroupCount - 1 + j] = proof;
        }

        for(let j = GroupCount - 2; j >= 0; j--) {
            let leftProof = proofs[2 * j + 1];
            let rightProof = proofs[2 * j + 2];
            let input = new GroupInput({
                rowId: leftProof.publicInput.rowId,
                groupIdStart: leftProof.publicInput.groupIdStart,
                groupIdEnd: rightProof.publicInput.groupIdEnd,
                zLeft: leftProof.publicInput.zLeft,
                zRight: rightProof.publicInput.zRight,
                z: leftProof.publicInput.z,
            });

            const proof = await Group.merge(input, leftProof, rightProof);

            const start = performance.now();
            const ok = await verify(proof.proof, vkGroup);
            const end = performance.now();

            console.log(`${nowPrefix()} Gate Group merge proof ${rowId} ${j} verify result: ${ok}, verifying time: ${end - start} ms`);
            proofs[j] = proof.proof;
        }

        let proofStr = JSON.stringify(proofs[0].toJSON())
        await fs.writeFile(`proofs/pos_${posId}/layer_${layerId}/gate_group/row_${rowId}/group_merge_0.json`, proofStr, "utf8");
    }

    class SortedGroupInput extends Struct({
        rowId: UInt32,
        groupIdStart: UInt32,
        groupIdEnd: UInt32,

        z: Field,
    }) {}

    class SortedGroupOutput extends Struct({
        largest2max: UInt64,
        largest2min: UInt64,
        cxs2: Field,
        cxs8: Field,
        cxs32: Field,
        zsumX: Field,
    }) {}

    const SortedGroup = ZkProgram({
        name: 'SortedGroup',
        publicInput: SortedGroupInput,
        publicOutput: SortedGroupOutput,
        methods: {
            base: {
                privateInputs: [GroupProof],
                async method(input: SortedGroupInput, proof: GroupProofType) {
                    proof.verify();

                    let pInput = proof.publicInput;
                    let pOutput = proof.publicOutput;

                    input.rowId.assertEquals(pInput.rowId);
                    input.groupIdEnd.assertEquals(input.groupIdStart.add(1));

                    input.z.assertEquals(pInput.z, 'z not equal');

                    let cxs8 = Provable.if(input.groupIdStart.lessThan(UInt32.from(4)), pOutput.cxs8, Field(1));

                    const out = new SortedGroupOutput({
                        largest2max: pOutput.largest2,
                        largest2min: pOutput.largest2,
                        cxs2: pOutput.cxs2,
                        cxs8,
                        cxs32: pOutput.cxs32,
                        zsumX: pOutput.zsumX,
                      });
                    return {publicOutput: out};
                }
            },
            merge: {
                privateInputs: [SelfProof, SelfProof],
                async method(input: SortedGroupInput,
                    leftProof: InstanceType<typeof SelfProof<SortedGroupInput, SortedGroupOutput> > ,
                    rightProof: InstanceType<typeof SelfProof<SortedGroupInput, SortedGroupOutput> >) {

                    leftProof.verify();
                    rightProof.verify();

                    let leftInput = leftProof.publicInput;
                    let leftOutput = leftProof.publicOutput;
                    let rightInput = rightProof.publicInput;
                    let rightOutput = rightProof.publicOutput;

                    input.rowId.assertEquals(leftInput.rowId);

                    input.groupIdStart.assertEquals(leftInput.groupIdStart);
                    input.groupIdEnd.assertEquals(rightInput.groupIdEnd);
                    leftInput.groupIdEnd.assertEquals(rightInput.groupIdStart);

                    input.z.assertEquals(leftInput.z, 'z 1 not equal');
                    input.z.assertEquals(rightInput.z, 'z 2 not equal');


                    leftOutput.largest2max.assertGreaterThanOrEqual(leftOutput.largest2min);
                    leftOutput.largest2min.assertGreaterThanOrEqual(rightOutput.largest2max);
                    rightOutput.largest2max.assertGreaterThanOrEqual(rightOutput.largest2min);

                    let cxs2 = leftOutput.cxs2.mul(rightOutput.cxs2);
                    let cxs8 = leftOutput.cxs8.mul(rightOutput.cxs8);
                    let cxs32 = leftOutput.cxs32.mul(rightOutput.cxs32);
                    let zsumX = leftOutput.zsumX.add(rightOutput.zsumX);

                    const out = new SortedGroupOutput({
                        largest2max: leftOutput.largest2max,
                        largest2min: rightOutput.largest2min,
                        cxs2, cxs8, cxs32, zsumX });
                    return {publicOutput: out};
                }
            }
        }
    })

    const SortedGroupProof = ZkProgram.Proof(SortedGroup);
    type SortedGroupProofType = InstanceType<typeof SortedGroupProof>;
    type SortedGroupProofJSON = ReturnType<InstanceType<typeof SortedGroupProof>["toJSON"]>;


    async function compileSortedGroupWithCache() {
        const cache = Cache.FileSystem(`./o1js-cache/Gate-SortedGroup`);

        await compileGroupWithCache();

        // let str = await SortedGroup.analyzeMethods();
        // console.log('Gate SortedGroup info: ', str);
    
        return SortedGroup.compile({cache: cache});
    }

    async function calcSortedGroupBase(name: string, posId: number, layerId: number, rowId: number, vkSorted: VerificationKey) {
        let zStr: string = await readFile('proofs/embed/hash');
        let z = Field(zStr);

        let proofs: GroupProofType[] = new Array(GroupCount);
        for(let j = 0; j < GroupCount; j++) {
            let proofFile = `proofs/pos_${posId}/layer_${layerId}/gate_group/row_${rowId}/group_base_${GroupCount - 1 + j}.json`;
            let proofStr = await readFile(proofFile);
            const proofJson = JSON.parse(proofStr) as GroupProofJSON;
            const proof = await GroupProof.fromJSON(proofJson);
            proofs[j] = proof;
        }

        const sortedProofs = proofs.slice().sort((a, b) => {
            const vala = a.publicOutput.largest2.toBigInt();
            const valb = b.publicOutput.largest2.toBigInt();
            if (vala < valb)
                return 1;
            else if (vala > valb)
                return -1;
            return 0;
        });

        for(let i = 0; i < GroupCount; i++) {
            let input = new SortedGroupInput({
                rowId: UInt32.from(rowId),
                groupIdStart: UInt32.from(i),
                groupIdEnd: UInt32.from(i+1),
                z,
            })

            const earlierProof = sortedProofs[i];

            const proof = await SortedGroup.base(input, earlierProof);

            const start = performance.now();
            const ok = await verify(proof.proof, vkSorted);
            const end = performance.now();

            let proofResStr = JSON.stringify(proof.proof.toJSON());
            fs.mkdir(`proofs/pos_${posId}/layer_${layerId}/gate_group/row_${rowId}`, { recursive: true }).then(() =>
                fs.writeFile(`proofs/pos_${posId}/layer_${layerId}/gate_group/row_${rowId}/sorted_base_${GroupCount - 1 + i}.json`, proofResStr, "utf8")
            ).catch(console.error);

            console.log(`${nowPrefix()} Gate SortedGroup base proof ${rowId} ${GroupCount - 1  + i} verify result: ${ok}, verifying time: ${end - start} ms`);
        }
    }


    async function calcSortedGroupMerge(name: string, posId: number, layerId: number, rowId: number, vkSorted: VerificationKey) {
        let proofs: SortedGroupProofType[] = new Array(2 * GroupCount - 1);
        for (let j = 0; j < GroupCount; j++) {
            let proofFile = `proofs/pos_${posId}/layer_${layerId}/gate_group/row_${rowId}/sorted_base_${GroupCount - 1 + j}.json`;
            let proofStr = await readFile(proofFile);
            const proofJson = JSON.parse(proofStr) as SortedGroupProofJSON;
            const proof = await SortedGroupProof.fromJSON(proofJson);
            proofs[GroupCount - 1 + j] = proof;
        }

        for(let j = GroupCount - 2; j >= 0; j--) {
            let leftProof = proofs[2 * j + 1];
            let rightProof = proofs[2 * j + 2];
            let input = new SortedGroupInput({
                rowId: leftProof.publicInput.rowId,
                groupIdStart: leftProof.publicInput.groupIdStart,
                groupIdEnd: rightProof.publicInput.groupIdEnd,
                z: leftProof.publicInput.z,
            });

            const proof = await SortedGroup.merge(input, leftProof, rightProof);

            const start = performance.now();
            const ok = await verify(proof.proof, vkSorted);
            const end = performance.now();

            console.log(`${nowPrefix()} Gate SortedGroup merge proof ${rowId} ${j} verify result: ${ok}, verifying time: ${end - start} ms`);
            proofs[j] = proof.proof;
        }

        let proofStr = JSON.stringify(proofs[0].toJSON())
        await fs.writeFile(`proofs/pos_${posId}/layer_${layerId}/gate_group/row_${rowId}/sorted_merge_0.json`, proofStr, "utf8");
    }

    class SelectorInput extends Struct({
        rowStart: UInt32,
        rowEnd: UInt32,
        zInRowStart: Field,
        zInRowEnd: Field,

        zOutRowStart: Field,
        zOutRowEnd: Field,

        z: Field,
    }) {}

    class SelectorOutput extends Struct({
        zsumX: Field,
        hashBias: Field,
        zsumY1: Field,
        zsumY2: Field,
    }) {}

    const CandidateUInt64 = Provable.Array(IdxValue, 8 * 4);

    const ExpertSelector = ZkProgram({
        name: 'ExpertSelector',
        publicInput: SelectorInput,
        publicOutput: SelectorOutput,
        methods: {
            base: {
                privateInputs: [GroupProof, SortedGroupProof, CandidateUInt64],
                async method(input: SelectorInput, groupProof: GroupProofType , sortedGroupProof: SortedGroupProofType, candidates: IdxValue[] ) {
                    groupProof.verify();
                    sortedGroupProof.verify();

                    let groupInput = groupProof.publicInput;
                    let groupOutput = groupProof.publicOutput;
                    let sortedGInput = sortedGroupProof.publicInput;
                    let sortedGOutput = sortedGroupProof.publicOutput;

                    input.rowStart.assertEquals(groupInput.rowId);
                    input.rowStart.assertEquals(sortedGInput.rowId);
                    input.rowEnd.assertEquals(input.rowStart.add(1));

                    input.zInRowStart.assertEquals(groupInput.zLeft, 'zInRowStart not match with groupInput.zLeft');
                    input.zInRowEnd.assertEquals(groupInput.zRight, 'zInRowEnd not match with groupInput.zRight');

                    input.z.assertEquals(groupInput.z, 'z 1 not equal');
                    input.z.assertEquals(sortedGInput.z, 'z 2 not equal');

                    groupInput.groupIdStart.assertEquals(UInt32.from(0));
                    groupInput.groupIdEnd.assertEquals(UInt32.from(GroupCount));

                    sortedGInput.groupIdStart.assertEquals(UInt32.from(0));
                    sortedGInput.groupIdEnd.assertEquals(UInt32.from(GroupCount));

                    groupOutput.cxs2.assertEquals(sortedGOutput.cxs2);
                    // groupOutput.cxs8.assertEquals(sortedGOutput.cxs8);
                    groupOutput.cxs32.assertEquals(sortedGOutput.cxs32);
                    groupOutput.zsumX.assertEquals(sortedGOutput.zsumX);

                    let z = input.z;
                    let c = z.neg();
                    let zi = input.zOutRowStart;

                    let cand0 = candidates[0];

                    let h = Poseidon.hash([cand0.idx.value, cand0.val0.toField(), cand0.val.value]);
                    let cxs = h.add(c);

                    // Provable.asProver(() => {
                    //     console.log(`cand[0]: ${cand0.idx.toBigint()}, ${cand0.val0.toBigint()}, ${cand0.val.toBigInt()}, ${h.toBigInt()}`);
                    // });

                    let zsumY1 = zi.mul(cand0.val.value);
                    let zsumY2 = zi.mul(cand0.idx.value);

                    zi = zi.mul(z);

                    for(let i = 1; i < 8; i++) {
                        let cand = candidates[i];
                        candidates[i-1].val.assertGreaterThanOrEqual(cand.val);
                        h = Poseidon.hash([cand.idx.value, cand.val0.toField(), cand.val.value]);
                        cxs = cxs.mul(h.add(c));

                        // Provable.asProver(() => {
                        //     console.log(`cand[${i}]: ${cand.idx.toBigint()}, ${cand.val0.toBigint()}, ${cand.val.toBigInt()}, ${h.toBigInt()}`);
                        // });

                        zsumY1 = zsumY1.add(zi.mul(cand.val.value));
                        zsumY2 = zsumY2.add(zi.mul(cand.idx.value));
                        zi = zi.mul(z);
                    }
                    input.zOutRowEnd.assertEquals(zi, 'zOutRowStart zOutRowEnd not match');

                    for(let i = 8; i < 8 * 4; i++) {
                        let cand = candidates[i];
                        candidates[i-1].val.assertGreaterThanOrEqual(cand.val);
                        let h = Poseidon.hash([cand.idx.value, cand.val0.toField(), cand.val.value]);
                        cxs = cxs.mul(h.add(c));

                        // Provable.asProver(() => {
                        //     console.log(`cand[${i}]: ${h.toBigInt()}`);
                        // });
                    }

                    cxs.assertEquals(sortedGOutput.cxs8);

                    const out = new SelectorOutput({
                        zsumX: groupOutput.zsumX,
                        hashBias: groupOutput.hashBias,
                        zsumY1,
                        zsumY2,
                    });
                    return {publicOutput: out};
                }
            },
            merge: {
                privateInputs: [SelfProof, SelfProof],
                async method(input: SelectorInput,
                    upProof: InstanceType<typeof SelfProof<SelectorInput, SelectorOutput> > ,
                    downProof: InstanceType<typeof SelfProof<SelectorInput, SelectorOutput> >) {

                    upProof.verify();
                    downProof.verify();

                    let upInput = upProof.publicInput;
                    let upOutput = upProof.publicOutput;
                    let downInput = downProof.publicInput;
                    let downOutput = downProof.publicOutput;

                    input.rowStart.assertEquals(upInput.rowStart);
                    input.rowEnd.assertEquals(downInput.rowEnd);
                    upInput.rowEnd.assertEquals(downInput.rowStart);


                    input.zInRowStart.assertEquals(upInput.zInRowStart, 'zInRowStart not equal');
                    input.zInRowEnd.assertEquals(downInput.zInRowEnd, 'zInRowEnd not equal');
                    upInput.zInRowEnd.assertEquals(downInput.zInRowStart, 'zInRowStart zInRowEnd not equal');

                    input.zOutRowStart.assertEquals(upInput.zOutRowStart, 'zOutRowStart not equal');
                    input.zOutRowEnd.assertEquals(downInput.zOutRowEnd, 'zOutRowEnd not equal');
                    upInput.zOutRowEnd.assertEquals(downInput.zOutRowStart, 'zOutRowStart zOutRowEnd not equal');


                    input.z.assertEquals(upInput.z, 'z 1 not equal');
                    input.z.assertEquals(downInput.z, 'z 2 not equal');

                    upOutput.hashBias.assertEquals(downOutput.hashBias, 'hashBias not equal');

                    let zsumX = upOutput.zsumX.add(downOutput.zsumX);
                    let zsumY1 = upOutput.zsumY1.add(downOutput.zsumY1);
                    let zsumY2 = upOutput.zsumY2.add(downOutput.zsumY2);

                    const out = new SelectorOutput({
                        zsumX,
                        hashBias: upOutput.hashBias,
                        zsumY1,
                        zsumY2,
                    });
                    return {publicOutput: out};
                }
            }
        }
    })

    const ExpertSelectorProof = ZkProgram.Proof(ExpertSelector);
    type ExpertSelectorProofType = InstanceType<typeof ExpertSelectorProof>;
    type ExpertSelectorProofJSON = ReturnType<InstanceType<typeof ExpertSelectorProof>["toJSON"]>;


    async function compileExpertSelectorWithCache() {
        const cache = Cache.FileSystem(`./o1js-cache/Gate-ExpertSelector`);

        await compileGroupWithCache();
        await compileSortedGroupWithCache();

        // let str = await ExpertSelector.analyzeMethods();
        // console.log('Gate ExpertSelector info: ', str);

        return ExpertSelector.compile({cache: cache});
    }

    async function selectorBase(name: string, posId: number, layerId: number, rowId: number, vkSelector: VerificationKey) {
        let Dim = GroupDim * GroupCount;

        let xs: Int64[][] = [];
        const bufX = await readBinary(`${zkDataDir}/pos_${posId}/layer_${layerId}/gate_original_scores.bin`);
        const xData = bufferToInt64ArrayLE(bufX);
        for(let i = 0; i < xData.length; i += Dim) {
            xs.push(xData.slice(i, i + Dim));
        }

        // let bs: Int64[] = [];
        const bufB = await readBinary(`${zkDataDir}/pos_${posId}/layer_${layerId}/gate_bias.bin`);
        const bs0 = bufferToUInt32ArrayLE(bufB);
        const bs = bs0.map(x => asInt64(x));

        let items: IdxValue[][] = [];
        for(let i = 0; i < xs.length; i++) {
            let row: IdxValue[] = [];
            for(let j = 0; j < Dim; j++) {
                let x = xs[i][j];
                let b = bs[j];
                // console.log(`x: ${x}, b:${b}`);
                let v = x.add(b);
                if(v < Int64.from(0)) {
                    console.log(`!!! scores[${i}][${j}] = ${v} < 0 !!!`);
                }
                let vv = UInt64.from(v.toBigint());
                let item = new IdxValue({
                    idx: UInt32.from(j),
                    val0: Int64.from(x),
                    bias: Int64.from(b),
                    val: vv,
                });
                row.push(item);
            }
            items.push(row);
        }

        let zStr: string = await readFile('proofs/embed/hash');
        let z = Field(zStr);

        let proofs: GroupProofType[] = new Array(GroupCount);
        for(let j = 0; j < GroupCount; j++) {
            let proofFile = `proofs/pos_${posId}/layer_${layerId}/gate_group/row_${rowId}/group_base_${GroupCount - 1 + j}.json`;
            let proofStr = await readFile(proofFile);
            const proofJson = JSON.parse(proofStr) as GroupProofJSON;
            const proof = await GroupProof.fromJSON(proofJson);
            proofs[j] = proof;
        }

        const sortedProofs = proofs.slice().sort((a, b) => {
            const vala = a.publicOutput.largest2.toBigInt();
            const valb = b.publicOutput.largest2.toBigInt();
            if (vala < valb)
                return 1;
            else if (vala > valb)
                return -1;
            return 0;
        });

        let candidates: IdxValue[] = [];
        for(let i = 0; i < 4; i++) {
            const groupId = Number(sortedProofs[i].publicInput.groupIdStart.toBigint());

            let left = groupId * GroupDim;
            let right = (groupId + 1)* GroupDim;

            // let largest2 = sortedProofs[i].publicOutput.largest2.toBigInt();
            // console.log(`groupId: ${groupId}, left: ${left}, ${right}, largest2: ${largest2}`);

            let xItems = items[rowId].slice(left, right);
            const sortedItems = xItems.slice().sort((a, b) => {
                const vala = a.val.toBigInt();
                const valb = b.val.toBigInt();
                if (vala < valb)
                    return 1;
                else if (vala > valb)
                    return -1;
                return 0;
            });

            for(let j = 0; j < 8; j++) {
                candidates.push(sortedItems[j]);
                // console.log(`${sortedItems[j].idx}, ${sortedItems[j].val}`);
            }
        }
        const sortedCandidates = candidates.slice().sort((a, b) => {
            const vala = a.val.toBigInt();
            const valb = b.val.toBigInt();
            if (vala < valb)
                return 1;
            else if (vala > valb)
                return -1;
            return 0;
        });

        let input = new SelectorInput({
            rowStart: UInt32.from(rowId),
            rowEnd: UInt32.from(rowId+1),
            zInRowStart: fastPow(z, rowId * Dim),
            zInRowEnd: fastPow(z, (rowId+1) * Dim),

            zOutRowStart: fastPow(z, rowId * 8),
            zOutRowEnd: fastPow(z, (rowId+1) * 8),

            z,
        });

        let proofFile1 = `proofs/pos_${posId}/layer_${layerId}/gate_group/row_${rowId}/group_merge_0.json`;
        let proofStr1 = await readFile(proofFile1);
        const proofJson1 = JSON.parse(proofStr1) as GroupProofJSON;
        const proof1 = await GroupProof.fromJSON(proofJson1);

        let proofFile2 = `proofs/pos_${posId}/layer_${layerId}/gate_group/row_${rowId}/sorted_merge_0.json`;
        let proofStr2 = await readFile(proofFile2);
        const proofJson2 = JSON.parse(proofStr2) as SortedGroupProofJSON;
        const proof2 = await SortedGroupProof.fromJSON(proofJson2);

        const proof = await ExpertSelector.base(input, proof1, proof2, sortedCandidates);

        const start = performance.now();
        const ok = await verify(proof.proof, vkSelector);
        const end = performance.now();

        console.log(`${nowPrefix()} ExpertSelector check proof ${rowId} verify result: ${ok}, verifying time: ${end - start} ms`);

        let proofStr = JSON.stringify(proof.proof.toJSON())
        await fs.writeFile(`proofs/pos_${posId}/layer_${layerId}/gate_group/row_${rowId}/check.json`, proofStr, "utf8");
    }

    async function selectorMerge(name: string, tokenListLen: number, posId: number, layerId: number, vkSelector: VerificationKey) {
        let rowsProofs: ExpertSelectorProofType[] = new Array(tokenListLen);
        for (let i = 0; i < tokenListLen; i++) {
            let proofFile = `proofs/pos_${posId}/layer_${layerId}/gate_group/row_${i}/check.json`;
            let proofStr = await readFile(proofFile);
            const proofJson = JSON.parse(proofStr) as ExpertSelectorProofJSON;
            const proof = await ExpertSelectorProof.fromJSON(proofJson);
            // const ok = await verify(proof, vkNorm); 
            // assert(ok, `base proof base_${rowId}_${j} verify failed!`);
            rowsProofs[i] = proof;
            // console.log(`add base_${ShortDimCount-1 + j}.json to proofs ${ShortDimCount - 1 + j}`);
        }

        while(rowsProofs.length > 1) {
            let rowsProofs2: ExpertSelectorProofType[]  = new Array(Math.floor((rowsProofs.length + 1) / 2));
            if(rowsProofs.length % 2 == 1) {
                rowsProofs2[rowsProofs2.length - 1] = rowsProofs[rowsProofs.length - 1];
                console.log(`add proofs2 ${rowsProofs2.length - 1} directly`);
            }

            for(let i = 0; i < Math.floor(rowsProofs.length / 2); i++) {
                let left = rowsProofs[2 * i];
                let right = rowsProofs[2 * i + 1];
                let input = new SelectorInput({
                    rowStart: left.publicInput.rowStart,
                    rowEnd: right.publicInput.rowEnd,
                    zInRowStart: left.publicInput.zInRowStart,
                    zInRowEnd: right.publicInput.zInRowEnd,

                    zOutRowStart: left.publicInput.zOutRowStart,
                    zOutRowEnd: right.publicInput.zOutRowEnd,

                    z: left.publicInput.z,
                });

                const proof = await ExpertSelector.merge(input, left, right);

                const start = performance.now();
                const ok = await verify(proof.proof, vkSelector);
                const end = performance.now();

                console.log(`${nowPrefix()} ${name} ExpertSelector merge proof ${i} verify result: ${ok}, verifying time: ${end - start} ms`);
                rowsProofs2[i] = proof.proof;
            }

            rowsProofs = rowsProofs2;
        }

        const proof = rowsProofs[0];
        let proofStr = JSON.stringify(proof.toJSON())
        await fs.writeFile(`proofs/pos_${posId}/layer_${layerId}/gate_group/expert_selector.json`, proofStr, "utf8");
    }

    return {Group, compileGroupWithCache, calcGroupBase, calcGroupMerge,
            SortedGroup, compileSortedGroupWithCache, calcSortedGroupBase, calcSortedGroupMerge,
            ExpertSelector, compileExpertSelectorWithCache, selectorBase, selectorMerge
        }
}

async function readFile(path: string): Promise<string> {
    const content = await fs.readFile(path, 'utf-8'); // 读取文件内容
    return content;
}

async function readJsonFile<T = any>(path: string): Promise<T> {
  const content = await fs.readFile(path, 'utf-8'); // 读取文件内容
  return JSON.parse(content); // 将内容解析为 JSON 对象
}


async function readAt(path: string, offset: number, length: number) {
  const fh = await fs.open(path, "r");
  try {
    const buf = Buffer.allocUnsafe(length);
    await fh.read(buf, 0, length, offset);   // 从 offset 读 length 字节
    return buf;
  } finally {
    await fh.close();
  }
}


function bufferToInt64ArrayLE(buf: Buffer): Int64[] {
  if (buf.length % 8 !== 0) {
    throw new RangeError("Buffer length must be multiple of 8");
  }
  const n = buf.length / 8;
  const arr = new Array<Int64>(n);
  for (let i = 0; i < n; i++) {
    arr[i] = Int64.from(buf.readBigInt64LE(i * 8)); // 小端；大端用 readBigInt64BE
  }
  return arr;
}

function bufferToUInt32ArrayLE(buf: Buffer): UInt32[] {
  if (buf.length % 4 !== 0) {
    throw new RangeError("Buffer length must be multiple of 4");
  }
  const n = buf.length / 4;
  const arr = new Array<UInt32>(n);
  for (let i = 0; i < n; i++) {
    arr[i] = UInt32.from(buf.readUInt32LE(i * 4)); // 小端；大端用 readBigInt64BE
  }
  return arr;
}

async function getAnEmbedFromFile(tokenId: bigint): Promise<Int64[]> {
    const fileId = BigInt(tokenId) / BigInt(EmbedsInOneFile);
    const filePath = EmbedsDir + fileId + '.bin';
    const rowId = BigInt(tokenId) % BigInt(EmbedsInOneFile);
    const rowOffset = BigInt(BytesOfAnEmbed) * rowId;
    // console.log('fileId: ' + fileId + ', rowId: ' + rowId);
    let embedbuf = await readAt(filePath, Number(rowOffset), BytesOfAnEmbed);
    let embed = bufferToInt64ArrayLE(embedbuf);
    // let embed0 = bufferToUInt32ArrayLE(embedbuf);
    // let xxs = embed.map(x => x.toBigint());
    // console.log(`xxs00: `, xxs);

    // let embed = embed0.map(x => asInt64(x));
    return embed;
}

async function readBinary(path: string): Promise<Buffer> {
  const buf = await fs.readFile(path); // 不要传编码参数 → 返回 Buffer（二进制）
  return buf;
}

function fieldToHex(f: Field, bytes = 32): string {
  const hex = f.toBigInt().toString(16);         // BigInt -> hex（大端，不含 0x）
  return '0x' + hex.padStart(bytes * 2, '0');    // 补到固定长度
}

const {RowSection, RowSectionProof, compileEmbedSection, sectionBase: embedSectionBase, sectionMerge: embedSectionMerge,
  Rows, RowsProof, compileEmbedRows, rowsMerge: embedRowsMerge, computeHash, precomputeHashes, computeEmbedHash} = createEmbedClass('embed', 7168, 224);

const { NormX: NormXAttn, NormXProof: NormXProofAttn, compileNormXWithCache: compileNormXWithCacheAttn,
  NormRows: NormRowsAttn, NormRowsProof: NormRowsProofAttn, compileNormRowsWithCache: compileNormRowsWithCacheAttn,
  calcBase: calcBaseAttn, calcMerge: calcMergeAttn, wrapRow: wrapRowAttn, mergeRow: mergeRowAttn } = createNormClass('attn_norm', 0n, 7168, 112);

const { NormX: NormXQ, NormXProof: NormXProofQ, compileNormXWithCache: compileNormXWithCacheQ,
    NormRows: NormRowsQ, NormRowsProof: NormRowsProofQ, compileNormRowsWithCache: compileNormRowsWithCacheQ,
    calcBase: calcBaseQ, calcMerge: calcMergeQ, wrapRow: wrapRowQ, mergeRow: mergeRowQ } = createNormClass('q_norm', 1n << 30n,  1536, 48);

const { GemmX: GemmX_Wqa, GemmXProof, compileGemmXWithCache: compileGemmXWithCache_Wqa, gemmXBase: gemmXBase_Wqa, gemmXMergeRow: gemmXMergeRow_Wqa,
  GemmW: GemmW_Wqa, GemmWProof, compileGemmWWithCache: compileGemmWWithCache_Wqa, gemmWBase: gemmWBase_Wqa, gemmWMergeRow: gemmWMergeRow_Wqa,
  GemmXW: GemmXW_Wqa, GemmXWProof, compileGemmXWWithCache: compileGemmXWWithCache_Wqa, gemmXWBase: gemmXWBase_Wqa, gemmXWMerge: gemmXWMerge_Wqa,
  checkGemm: CheckGemm_Wqa }
  = createGemmClass('wq_a', 7168, 1536, 112)

const  { Rope, RopeProof, compileRopeWithCache, calcRopeBase, calcRopeMerge,
  RopeRows, RopeRowsProof, compileRopeRowsWithCache, wrapRopeRow, mergeRopeRow,
  calcRopeHashes } = createRopeClass('q_pe');

const { GemmX: GemmX_Wkv_a1, GemmXProof: GemmXProof_Wkv_a1, compileGemmXWithCache: compileGemmXWithCache_Wkv_a1, gemmXBase: gemmXBase_Wkv_a1, gemmXMergeRow: gemmXMergeRow_Wkv_a1,
    GemmW: GemmW_Wkv_a1, GemmWProof: GemmWProof_Wkv_a1, compileGemmWWithCache: compileGemmWWithCache_Wkv_a1, gemmWBase: gemmWBase_Wkv_a1, gemmWMergeRow: gemmWMergeRow_Wkv_a1,
    GemmXW: GemmXW_Wkv_a1, GemmXWProof: GemmXWProof_Wkv_a1, compileGemmXWWithCache: compileGemmXWWithCache_Wkv_a1, gemmXWBase: gemmXWBase_Wkv_a1, gemmXWMerge: gemmXWMerge_Wkv_a1,
    checkGemm: CheckGemm_Wkv_a1 }
    = createGemmClass('wkv_a1', 7168, 512, 112)

const { Head, compileSoftmaxHeadWithCache, calcHeadBase: softmaxHeadBase, calcHeadMerge: softmaxHeadMerge,
      Rows: SoftmaxRows, compileSoftmaxRowsWithCache, softmaxWrapRow, softmaxMergeRow } = createSoftmaxClass('scores');

const {Section, compileSigmoidSectionWithCache: compileSigmoidSection, calcSectionBase: sigmoidSectionBase, calcSectionMerge: sigmoidSectionMerge,
        Rows: SigmoidRows, compileSigmoidRowsWithCache: compileSigmoidRows, calcRowsBase: sigmoidRowBase, calcRowMerge: sigmoidRowMerge } = createSigmoidClass('gate', 256, 16);


const {Group, compileGroupWithCache: compileExpertsGroup, calcGroupBase: expertsGroupBase, calcGroupMerge: expertsGroupMerge,
          SortedGroup, compileSortedGroupWithCache: compileSortedGroup, calcSortedGroupBase: expertsSortedGroupBase, calcSortedGroupMerge: expertsSortedGroupMerge,
          ExpertSelector, compileExpertSelectorWithCache: compileExpertSelector, selectorBase: expertsSelectorBase, selectorMerge: expertsSelectorMerge
      } = createExpertsSelectorClass('gate', 32, 8);


async function getAllTokenIds(){
    const bufX = await readBinary(`${zkDataDir}/pos_0/tokens.bin`);
    const xs = bufferToInt64ArrayLE(bufX);
    let allTokenIds = xs.map(x => Number(x.toBigint()))
    return allTokenIds;
}


async function main() {
  const program = new Command();

  program
    .name("zk-cli")
    .description("ZK pipeline commands")
    .version("0.1.0");

  // ------------------- embedSectionBase / embedSectionMerge -------------------
  program
    .command("embedSectionBase")
    .argument("<name>", "component name")
    .argument("<rowId>", "row id")
    .description("Run embedSectionBase")
    .action(async (name: string, rowIdStr: string) => {
      const rowId = Number(rowIdStr);
      const allTokenIds = await getAllTokenIds();
      const { verificationKey: vk } = await compileEmbedSection();

      console.log(`${nowPrefix()} embedSectionBase --- `);
      await embedSectionBase(name, allTokenIds, rowId, vk);
    });

  program
    .command("embedSectionMerge")
    .argument("<name>", "component name")
    .argument("<rowId>", "row id")
    .description("Run embedSectionMerge")
    .action(async (name: string, rowIdStr: string) => {
      const rowId = Number(rowIdStr);
      const { verificationKey: vk } = await compileEmbedSection();

      console.log(`${nowPrefix()} embedSectionMerge --- `);
      await embedSectionMerge(name, rowId, vk);
    });

  // ------------------- embedRowsMerge -------------------
  program
    .command("embedRowsMerge")
    .argument("<name>", "component name")
    .description("Run embedRowsMerge")
    .action(async (name: string) => {
      const allTokenIds = await getAllTokenIds();
      const { verificationKey: vk } = await compileEmbedRows();

      console.log(`${nowPrefix()} embedRowsMerge --- `);
      await embedRowsMerge(name, allTokenIds.length, vk);
    });

  // ------------------- computeHash / precomputeHashes / computeEmbedHash -------------------
  program
    .command("computeHash")
    .argument("<name>", "component name (unused)")
    .argument("<tokenId>", "token id")
    .description("Compute hash for a single token")
    .action(async (_name: string, tokenIdStr: string) => {
      const tokenId = Number(tokenIdStr);
      // console.log(`${nowPrefix()} computeHash --- `);
      await computeHash(BigInt(tokenId));
    });

  program
    .command("precomputeHashes")
    .argument("<name>", "component name (unused)")
    .description("Precompute hashes")
    .action(async (_name: string) => {
      console.log(`${nowPrefix()} precomputeHashes --- `);
      await precomputeHashes();
    });

  program
    .command("computeEmbedHash")
    .argument("<name>", "component name (unused)")
    .description("Compute embedding hash for all input tokens")
    .action(async (_name: string) => {
      const allTokenIds = await getAllTokenIds();
      console.log(`${nowPrefix()} computeEmbedHash --- `);
      await computeEmbedHash(allTokenIds);
    });

  // ------------------- normBase / normMerge -------------------
  program
    .command("normBase")
    .argument("<name>", "attn_norm | q_norm")
    .argument("<posId>", "position id")
    .argument("<layerId>", "layer id")
    .argument("<rowId>", "row id")
    .argument("<ind>", "index")
    .description("Run normBase")
    .action(async (name: string, posIdStr: string, layerIdStr: string, rowIdStr: string, indStr: string) => {
      const posId = Number(posIdStr);
      const layerId = Number(layerIdStr);
      const rowId = Number(rowIdStr);
      const ind = Number(indStr);

      if (name === "attn_norm") {
        const { verificationKey: vk } = await compileNormXWithCacheAttn();
        console.log(`${nowPrefix()} normBase --- `);
        await calcBaseAttn(name, posId, layerId, rowId, ind, vk);
      } else if (name === "q_norm") {
        const { verificationKey: vk } = await compileNormXWithCacheQ();
        console.log(`${nowPrefix()} normBase --- `);
        await calcBaseQ(name, posId, layerId, rowId, ind, vk);
      } else {
        console.error(`Unknown norm name: ${name}`);
      }
    });

  program
    .command("normMerge")
    .argument("<name>", "attn_norm | q_norm")
    .argument("<posId>", "position id")
    .argument("<layerId>", "layer id")
    .argument("<rowId>", "row id")
    .argument("<ind>", "index")
    .description("Run normMerge")
    .action(async (name: string, posIdStr: string, layerIdStr: string, rowIdStr: string, indStr: string) => {
      const posId = Number(posIdStr);
      const layerId = Number(layerIdStr);
      const rowId = Number(rowIdStr);
      const ind = Number(indStr);

      if (name === "attn_norm") {
        const { verificationKey: vk } = await compileNormXWithCacheAttn();
        console.log(`${nowPrefix()} normMerge --- `);
        await calcMergeAttn(name, posId, layerId, rowId, ind, vk);
      } else if (name === "q_norm") {
        const { verificationKey: vk } = await compileNormXWithCacheQ();
        console.log(`${nowPrefix()} normMerge --- `);
        await calcMergeQ(name, posId, layerId, rowId, ind, vk);
      } else {
        console.error(`Unknown norm name: ${name}`);
      }
    });

  // ------------------- normWrapRow / normMergeRow -------------------
  program
    .command("normWrapRow")
    .argument("<name>", "attn_norm | q_norm")
    .argument("<posId>", "position id")
    .argument("<layerId>", "layer id")
    .description("Run normWrapRow")
    .action(async (name: string, posIdStr: string, layerIdStr: string) => {
      const posId = Number(posIdStr);
      const layerId = Number(layerIdStr);
      const allTokenIds = await getAllTokenIds();

      if (name === "attn_norm") {
        const { verificationKey: vk } = await compileNormRowsWithCacheAttn();
        console.log(`${nowPrefix()} normWrapRow --- `);
        await wrapRowAttn(name, allTokenIds.length, posId, layerId, vk);
      } else if (name === "q_norm") {
        const { verificationKey: vk } = await compileNormRowsWithCacheQ();
        console.log(`${nowPrefix()} normWrapRow --- `);
        await wrapRowQ(name, allTokenIds.length, posId, layerId, vk);
      } else {
        console.error(`Unknown norm name: ${name}`);
      }
    });

  program
    .command("normMergeRow")
    .argument("<name>", "attn_norm | q_norm")
    .argument("<posId>", "position id")
    .argument("<layerId>", "layer id")
    .description("Run normMergeRow")
    .action(async (name: string, posIdStr: string, layerIdStr: string) => {
      const posId = Number(posIdStr);
      const layerId = Number(layerIdStr);
      const allTokenIds = await getAllTokenIds();

      if (name === "attn_norm") {
        const { verificationKey: vk } = await compileNormRowsWithCacheAttn();
        console.log(`${nowPrefix()} normMergeRow --- `);
        await mergeRowAttn(name, allTokenIds.length, posId, layerId, vk);
      } else if (name === "q_norm") {
        const { verificationKey: vk } = await compileNormRowsWithCacheQ();
        console.log(`${nowPrefix()} normMergeRow --- `);
        await mergeRowQ(name, allTokenIds.length, posId, layerId, vk);
      } else {
        console.error(`Unknown norm name: ${name}`);
      }
    });

  // ------------------- gemmXBase / gemmXMergeRow -------------------
  program
    .command("gemmXBase")
    .argument("<name>", "wq_a | wkv_a1")
    .argument("<posId>", "position id")
    .argument("<layerId>", "layer id")
    .argument("<rowId>", "row id")
    .argument("<ind>", "index")
    .description("Run gemmXBase")
    .action(async (name: string, posIdStr: string, layerIdStr: string, rowIdStr: string, indStr: string) => {
      const posId = Number(posIdStr);
      const layerId = Number(layerIdStr);
      const rowId = Number(rowIdStr);
      const ind = Number(indStr);

      if (name === "wq_a") {
        const { verificationKey: vk } = await compileGemmXWithCache_Wqa();

        console.log(`${nowPrefix()} gemmXBase_Wqa --- `);
        await gemmXBase_Wqa(name, posId, layerId, rowId, ind, vk);
      } else if (name === "wkv_a1") {
        const { verificationKey: vk } = await compileGemmXWithCache_Wkv_a1();

        console.log(`${nowPrefix()} gemmXBase_Wkv_a1 --- `);
        await gemmXBase_Wkv_a1(name, posId, layerId, rowId, ind, vk);
      } else {
        console.error(`Unknown gemmX name: ${name}`);
      }
    });

  program
    .command("gemmXMergeRow")
    .argument("<name>", "wq_a | wkv_a1")
    .argument("<posId>", "position id")
    .argument("<layerId>", "layer id")
    .argument("<ind>", "index")
    .description("Run gemmXMergeRow")
    .action(async (name: string, posIdStr: string, layerIdStr: string, indStr: string) => {
      const posId = Number(posIdStr);
      const layerId = Number(layerIdStr);
      const ind = Number(indStr);
      const allTokenIds = await getAllTokenIds();

      if (name === "wq_a") {
        const { verificationKey: vk } = await compileGemmXWithCache_Wqa();

        console.log(`${nowPrefix()} gemmXMergeRow_Wqa --- `);
        await gemmXMergeRow_Wqa(name, allTokenIds.length, posId, layerId, ind, vk);
      } else if (name === "wkv_a1") {
        const { verificationKey: vk } = await compileGemmXWithCache_Wkv_a1();

        console.log(`${nowPrefix()} gemmXMergeRow_Wkv_a1 --- `);
        await gemmXMergeRow_Wkv_a1(name, allTokenIds.length, posId, layerId, ind, vk);
      } else {
        console.error(`Unknown gemmX name: ${name}`);
      }
    });

  // ------------------- gemmWBase / gemmWMergeRow -------------------
  program
    .command("gemmWBase")
    .argument("<name>", "wq_a | wkv_a1")
    .argument("<posId>", "position id")
    .argument("<layerId>", "layer id")
    .argument("<rowId>", "row id")
    .argument("<ind>", "index")
    .description("Run gemmWBase")
    .action(async (name: string, posIdStr: string, layerIdStr: string, rowIdStr: string, indStr: string) => {
      const posId = Number(posIdStr);
      const layerId = Number(layerIdStr);
      const rowId = Number(rowIdStr);
      const ind = Number(indStr);
      const allTokenIds = await getAllTokenIds();

      if (name === "wq_a") {
        const { verificationKey: vk } = await compileGemmWWithCache_Wqa();

        console.log(`${nowPrefix()} gemmWBase_Wqa --- `);
        await gemmWBase_Wqa(name, allTokenIds.length, posId, layerId, rowId, ind, vk);
      } else if (name === "wkv_a1") {
        const { verificationKey: vk } = await compileGemmWWithCache_Wkv_a1();

        console.log(`${nowPrefix()} gemmWBase_Wkv_a1 --- `);
        await gemmWBase_Wkv_a1(name, allTokenIds.length, posId, layerId, rowId, ind, vk);
      } else {
        console.error(`Unknown gemmW name: ${name}`);
      }
    });

  program
    .command("gemmWMergeRow")
    .argument("<name>", "wq_a | wkv_a1")
    .argument("<posId>", "position id")
    .argument("<layerId>", "layer id")
    .argument("<ind>", "index")
    .argument("<rowIndex>", "row index")
    .description("Run gemmWMergeRow")
    .action(async (name: string, posIdStr: string, layerIdStr: string, indStr: string, rowIndexStr: string) => {
      const posId = Number(posIdStr);
      const layerId = Number(layerIdStr);
      const ind = Number(indStr);
      const rowIndex = Number(rowIndexStr);
      const allTokenIds = await getAllTokenIds();

      if (name === "wq_a") {
        const { verificationKey: vk } = await compileGemmWWithCache_Wqa();

        console.log(`${nowPrefix()} gemmWMergeRow_Wqa --- `);
        await gemmWMergeRow_Wqa(name, allTokenIds.length, posId, layerId, ind, rowIndex, vk);
      } else if (name === "wkv_a1") {
        const { verificationKey: vk } = await compileGemmWWithCache_Wkv_a1();

        console.log(`${nowPrefix()} gemmWMergeRow_Wkv_a1 --- `);
        await gemmWMergeRow_Wkv_a1(name, allTokenIds.length, posId, layerId, ind, rowIndex, vk);
      } else {
        console.error(`Unknown gemmW name: ${name}`);
      }
    });

  // ------------------- gemmXWBase / gemmXWMerge -------------------
  program
    .command("gemmXWBase")
    .argument("<name>", "wq_a")
    .argument("<posId>", "position id")
    .argument("<layerId>", "layer id")
    .argument("<ind>", "index")
    .description("Run gemmXWBase")
    .action(async (name: string, posIdStr: string, layerIdStr: string, indStr: string) => {
      const posId = Number(posIdStr);
      const layerId = Number(layerIdStr);
      const ind = Number(indStr);
      const allTokenIds = await getAllTokenIds();

      if (name === "wq_a") {
        const { verificationKey: vk } = await compileGemmXWWithCache_Wqa();

        console.log(`${nowPrefix()} gemmXWBase_Wqa --- `);
        await gemmXWBase_Wqa(name, allTokenIds.length, posId, layerId, ind, vk);
      } else if (name === "wkv_a1") {
        const { verificationKey: vk } = await compileGemmXWWithCache_Wkv_a1();

        console.log(`${nowPrefix()} gemmXWBase_Wkv_a1 --- `);
        await gemmXWBase_Wkv_a1(name, allTokenIds.length, posId, layerId, ind, vk);
      } else {
        console.error(`Unknown gemmXW name: ${name}`);
      }
    });

  program
    .command("gemmXWMerge")
    .argument("<name>", "wq_a")
    .argument("<posId>", "position id")
    .argument("<layerId>", "layer id")
    .argument("<ind>", "index")
    .description("Run gemmXWMerge")
    .action(async (name: string, posIdStr: string, layerIdStr: string, indStr: string) => {
      const posId = Number(posIdStr);
      const layerId = Number(layerIdStr);
      const ind = Number(indStr);
      const allTokenIds = await getAllTokenIds();

      if (name === "wq_a") {
        const { verificationKey: vk } = await compileGemmXWWithCache_Wqa();

        console.log(`${nowPrefix()} gemmXWMerge_Wqa --- `);
        await gemmXWMerge_Wqa(name, allTokenIds.length, posId, layerId, ind, vk);
      } else if (name === "wkv_a1") {
        const { verificationKey: vk } = await compileGemmXWWithCache_Wkv_a1();

        console.log(`${nowPrefix()} gemmXWMerge_Wkv_a1 --- `);
        await gemmXWMerge_Wkv_a1(name, allTokenIds.length, posId, layerId, ind, vk);
      } else {
        console.error(`Unknown gemmXW name: ${name}`);
      }
    });

  // ------------------- checkGemm -------------------
  program
    .command("checkGemm")
    .argument("<name>", "wq_a")
    .description("Check GEMM")
    .action(async (_name: string) => {
      // await CheckGemm_Wqa();
      await CheckGemm_Wkv_a1();
      console.log(`${nowPrefix()} Finish!`);
    });

  // ------------------- ropeBase / ropeMerge -------------------
  program
    .command("ropeBase")
    .argument("<name>", "component name")
    .argument("<posId>", "position id")
    .argument("<layerId>", "layer id")
    .argument("<rowId>", "row id")
    .argument("<headId>", "head id")
    .description("Run ropeBase")
    .action(async (name: string, posIdStr: string, layerIdStr: string, rowIdStr: string, headIdStr: string) => {
      const posId = Number(posIdStr);
      const layerId = Number(layerIdStr);
      const rowId = Number(rowIdStr);
      const headId = Number(headIdStr);
      const { verificationKey: vk } = await compileRopeWithCache();

      console.log(`${nowPrefix()} ropeBase --- `);
      await calcRopeBase(name, posId, layerId, rowId, headId, vk);
    });

  program
    .command("ropeMerge")
    .argument("<name>", "component name")
    .argument("<posId>", "position id")
    .argument("<layerId>", "layer id")
    .argument("<rowId>", "row id")
    .argument("<ind>", "index")
    .description("Run ropeMerge")
    .action(async (name: string, posIdStr: string, layerIdStr: string, rowIdStr: string, indStr: string) => {
      const posId = Number(posIdStr);
      const layerId = Number(layerIdStr);
      const rowId = Number(rowIdStr);
      const ind = Number(indStr);
      const { verificationKey: vkRope } = await compileRopeWithCache();

      console.log(`${nowPrefix()} ropeMerge --- `);
      await calcRopeMerge(name, posId, layerId, rowId, ind, vkRope);
    });

  // ------------------- wrapRopeRow / mergeRopeRow -------------------
  program
    .command("wrapRopeRow")
    .argument("<name>", "component name (unused)")
    .argument("<posId>", "position id")
    .argument("<layerId>", "layer id")
    .description("Run wrapRopeRow")
    .action(async (_name: string, posIdStr: string, layerIdStr: string) => {
      const posId = Number(posIdStr);
      const layerId = Number(layerIdStr);
      const allTokenIds = await getAllTokenIds();
      const { verificationKey: vk } = await compileRopeRowsWithCache();

      console.log(`${nowPrefix()} wrapRopeRow --- `);
      await wrapRopeRow(allTokenIds.length, posId, layerId, vk);
    });

  program
    .command("mergeRopeRow")
    .argument("<name>", "component name (unused)")
    .argument("<posId>", "position id")
    .argument("<layerId>", "layer id")
    .description("Run mergeRopeRow")
    .action(async (_name: string, posIdStr: string, layerIdStr: string) => {
      const posId = Number(posIdStr);
      const layerId = Number(layerIdStr);
      const allTokenIds = await getAllTokenIds();
      const { verificationKey: vk } = await compileRopeRowsWithCache();

      console.log(`${nowPrefix()} mergeRopeRow --- `);
      await mergeRopeRow(allTokenIds.length, posId, layerId, vk);
    });

     // ------------------- softmaxHeadBase / softmaxHeadMerge -------------------
  program
  .command("softmaxHeadBase")
  .argument("<name>", "scores")
  .argument("<posId>", "position id")
  .argument("<layerId>", "layer id")
  .argument("<rowId>", "row id")
  .argument("<ind>", "index")
  .argument("<headDim>", "head dimension")
  .description("Run softmaxHeadBase")
  .action(async (name: string, posIdStr: string, layerIdStr: string, rowIdStr: string, indStr: string, headDimStr: string) => {
    const posId = Number(posIdStr);
    const layerId = Number(layerIdStr);
    const rowId = Number(rowIdStr);
    const ind = Number(indStr);
    const headDim = Number(headDimStr);

    const {vkSection, vkHead} = await compileSoftmaxHeadWithCache();

    console.log(`${nowPrefix()} softmaxHeadBase --- `);
    for(let j = ind; j < ind + 4; j++) {
      await softmaxHeadBase(name, posId, layerId, rowId, j, headDim, vkSection, vkHead);
    }
  });

  program
  .command("softmaxHeadMerge")
  .argument("<name>", "scores")
  .argument("<posId>", "position id")
  .argument("<layerId>", "layer id")
  .argument("<rowId>", "row id")
  .argument("<ind>", "index")
  .argument("<headDim>", "head dimension")
  .description("Run softmaxHeadMerge")
  .action(async (name: string, posIdStr: string, layerIdStr: string, rowIdStr: string, indStr: string, headDimStr: string) => {
    const posId = Number(posIdStr);
    const layerId = Number(layerIdStr);
    const rowId = Number(rowIdStr);
    const ind = Number(indStr);
    const headDim = Number(headDimStr);

    const {vkSection, vkHead} = await compileSoftmaxHeadWithCache();

    console.log(`${nowPrefix()} softmaxHeadMerge --- `);
    softmaxHeadMerge(name, posId, layerId, rowId, ind, headDim, vkHead);
  });

// ------------------- wrapRopeRow / mergeRopeRow -------------------
  program
  .command("softmaxWrapRow")
  .argument("<name>", "component name (unused)")
  .argument("<posId>", "position id")
  .argument("<layerId>", "layer id")
  .description("Run softmaxWrapRow")
  .action(async (_name: string, posIdStr: string, layerIdStr: string) => {
    const posId = Number(posIdStr);
    const layerId = Number(layerIdStr);
    const allTokenIds = await getAllTokenIds();

    const { verificationKey: vkRows } = await compileSoftmaxRowsWithCache();

    console.log(`${nowPrefix()} softmaxWrapRow --- `);

    await softmaxWrapRow('scores', allTokenIds.length, posId, layerId, vkRows)
  });

  program
  .command("softmaxMergeRow")
  .argument("<name>", "component name (unused)")
  .argument("<posId>", "position id")
  .argument("<layerId>", "layer id")
  .description("Run softmaxMergeRow")
  .action(async (_name: string, posIdStr: string, layerIdStr: string) => {
    const posId = Number(posIdStr);
    const layerId = Number(layerIdStr);
    const allTokenIds = await getAllTokenIds();

    const { verificationKey: vkRows } = await compileSoftmaxRowsWithCache();

    console.log(`${nowPrefix()} softmaxMergeRow --- `);

    await softmaxMergeRow('scores', allTokenIds.length, posId, layerId, vkRows)
  });

     // ------------------- sigmoidSectionBase / sigmoidSectionMerge -------------------
    program
     .command("sigmoidSectionBase")
     .argument("<name>", "gate")
     .argument("<posId>", "position id")
     .argument("<layerId>", "layer id")
     .argument("<rowId>", "row id")
     .description("Run sigmoidSectionBase")
     .action(async (name: string, posIdStr: string, layerIdStr: string, rowIdStr: string) => {
       const posId = Number(posIdStr);
       const layerId = Number(layerIdStr);
       const rowId = Number(rowIdStr);

       const {verificationKey: vk} = await compileSigmoidSection();

       console.log(`${nowPrefix()} sigmoidSectionBase --- `);
       await sigmoidSectionBase('gate', posId, layerId, rowId, vk)
     });

    program
     .command("sigmoidSectionMerge")
     .argument("<name>", "gate")
     .argument("<posId>", "position id")
     .argument("<layerId>", "layer id")
     .argument("<rowId>", "row id")
     .description("Run sigmoidSectionMerge")
     .action(async (name: string, posIdStr: string, layerIdStr: string, rowIdStr: string) => {
       const posId = Number(posIdStr);
       const layerId = Number(layerIdStr);
       const rowId = Number(rowIdStr);

       const {verificationKey: vk} = await compileSigmoidSection();

       console.log(`${nowPrefix()} sigmoidSectionMerge --- `);
       await sigmoidSectionMerge('gate', posId, layerId, rowId, vk)
     });

   // ------------------- sigmoidRowBase / sigmoidRowMerge -------------------
    program
     .command("sigmoidRowBase")
     .argument("<name>", "gate")
     .argument("<posId>", "position id")
     .argument("<layerId>", "layer id")
     .description("Run sigmoidRowBase")
     .action(async (_name: string, posIdStr: string, layerIdStr: string) => {
       const posId = Number(posIdStr);
       const layerId = Number(layerIdStr);
       const allTokenIds = await getAllTokenIds();

       const { verificationKey: vk } = await compileSigmoidRows();

       console.log(`${nowPrefix()} sigmoidRowBase --- `);
       await sigmoidRowBase('gate', allTokenIds.length, posId, layerId, vk)
     });

    program
     .command("sigmoidRowMerge")
     .argument("<name>", "gate")
     .argument("<posId>", "position id")
     .argument("<layerId>", "layer id")
     .description("Run sigmoidRowMerge")
     .action(async (name: string, posIdStr: string, layerIdStr: string) => {
       const posId = Number(posIdStr);
       const layerId = Number(layerIdStr);
       const allTokenIds = await getAllTokenIds();

       const { verificationKey: vk } = await compileSigmoidRows();

       console.log(`${nowPrefix()} sigmoidRowMerge --- `);
       await sigmoidRowMerge('gate', allTokenIds.length, posId, layerId, vk)
     });

      // ------------------- expertsGroupBase / expertsGroupMerge -------------------
    program
    .command("expertsGroupBase")
    .argument("<name>", "gate")
    .argument("<posId>", "position id")
    .argument("<layerId>", "layer id")
    .argument("<rowId>", "row id")
    .description("Run expertsGroupBase")
    .action(async (name: string, posIdStr: string, layerIdStr: string, rowIdStr: string) => {
      const posId = Number(posIdStr);
      const layerId = Number(layerIdStr);
      const rowId = Number(rowIdStr);

      const {verificationKey: vk} = await compileExpertsGroup();

      console.log(`${nowPrefix()} expertsGroupBase --- `);
      await expertsGroupBase('gate', posId, layerId, rowId, vk)
    });

    program
    .command("expertsGroupMerge")
    .argument("<name>", "gate")
    .argument("<posId>", "position id")
    .argument("<layerId>", "layer id")
    .argument("<rowId>", "row id")
    .description("Run expertsGroupMerge")
    .action(async (name: string, posIdStr: string, layerIdStr: string, rowIdStr: string) => {
      const posId = Number(posIdStr);
      const layerId = Number(layerIdStr);
      const rowId = Number(rowIdStr);

      const {verificationKey: vk} = await compileExpertsGroup();

      console.log(`${nowPrefix()} expertsGroupMerge --- `);
      await expertsGroupMerge('gate', posId, layerId, rowId, vk)
    });

  // ------------------- expertsSortedGroupBase / expertsSortedGroupMerge -------------------
   program
    .command("expertsSortedGroupBase")
    .argument("<name>", "gate")
    .argument("<posId>", "position id")
    .argument("<layerId>", "layer id")
    .argument("<rowId>", "row id")
    .description("Run expertsSortedGroupBase")
    .action(async (_name: string, posIdStr: string, layerIdStr: string, rowIdStr: string) => {
      const posId = Number(posIdStr);
      const layerId = Number(layerIdStr);
      const rowId = Number(rowIdStr);

      const { verificationKey: vk } = await compileSortedGroup();

      console.log(`${nowPrefix()} expertsSortedGroupBase --- `);
      await expertsSortedGroupBase('gate', posId, layerId, rowId, vk)
    });

    program
    .command("expertsSortedGroupMerge")
    .argument("<name>", "gate")
    .argument("<posId>", "position id")
    .argument("<layerId>", "layer id")
    .argument("<rowId>", "row id")
    .description("Run expertsSortedGroupMerge")
    .action(async (_name: string, posIdStr: string, layerIdStr: string, rowIdStr: string) => {
      const posId = Number(posIdStr);
      const layerId = Number(layerIdStr);
      const rowId = Number(rowIdStr);

      const { verificationKey: vk } = await compileSortedGroup();

      console.log(`${nowPrefix()} expertsSortedGroupMerge --- `);
      await expertsSortedGroupMerge('gate', posId, layerId, rowId, vk)
    });

    // ------------------- expertsSelectorBase / expertsSelectorMerge -------------------
   program
   .command("expertsSelectorBase")
   .argument("<name>", "gate")
   .argument("<posId>", "position id")
   .argument("<layerId>", "layer id")
   .argument("<rowId>", "row id")
   .description("Run expertsSelectorBase")
   .action(async (_name: string, posIdStr: string, layerIdStr: string, rowIdStr: string) => {
     const posId = Number(posIdStr);
     const layerId = Number(layerIdStr);
     const rowId = Number(rowIdStr);

     const { verificationKey: vk } = await compileExpertSelector();

     console.log(`${nowPrefix()} expertsSelectorBase --- `);
     await expertsSelectorBase('gate', posId, layerId, rowId, vk)
   });

   program
   .command("expertsSelectorMerge")
   .argument("<name>", "gate")
   .argument("<posId>", "position id")
   .argument("<layerId>", "layer id")
   .description("Run expertsSelectorMerge")
   .action(async (_name: string, posIdStr: string, layerIdStr: string) => {
     const posId = Number(posIdStr);
     const layerId = Number(layerIdStr);
     const allTokenIds = await getAllTokenIds();

     const { verificationKey: vk } = await compileExpertSelector();

     console.log(`${nowPrefix()} expertsSelectorMerge --- `);
     await expertsSelectorMerge('gate', allTokenIds.length, posId, layerId, vk)
   });


  // 解析命令行
  await program.parseAsync(process.argv);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});