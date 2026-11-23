using PyCall
using Arpack
using Plots
using TensorOperations
using LinearAlgebra
using LinearMaps
using NLsolve
using IterativeSolvers
using KrylovKit

include("amplitudes.jl")
include("jacobians.jl")

pyscf = pyimport("pyscf")
pyscf.lib.misc.num_threads(48)
cubegen = pyimport("pyscf.tools.cubegen")



function energy_pb_cc(γo, γv, t1s, t2s, λ, λdag)
       no = size(γo, 1)
       nv = size(γv, 1)
       λo = @view λ[1:no]
       λv = @view λ[(no+1):(no+nv)]
       λdago = @view λdag[1:no]
       λdagv = @view λdag[(1+no):(no+nv)]
	Ehf = 0
	Epb = 0
	Ecc1 = 0
	Ecc2 = 0
	@tensoropt begin
		Ehf += 2 * F_oo[i,i]
		Epb -= 2 * γo[i] * λdago[i]
		Epb += 2 * γv[a] * λv[a]
		Ehf -= L_oooo[i,i,j,j]
		Ecc1 += 2 * F_ov[i,a] * t1s[a,i]
		Ecc2 += L_ovov[i,a,j,b] * t2s[a,i,b,j]
		Ecc1 += L_ovov[i,a,j,b] * t1s[a,i] * t1s[b,j]
	end
	println("pb contribution: ", Epb)
	println("singles contribution: ", Ecc1)
	println("doubles contribution: ", Ecc2)
	Ehf, Epb, Ecc1, Ecc2
end


function make_pbcc_E_func(nocc, nvir, λ, λdag)
	function e_func(ω)
		γo, γv, t1s, t2s = parameters_untangle(nocc, nvir, ω)
        energy_pb_cc(γo, γv, t1s, t2s, λ, λdag)
    end
end

function symmetric_sum!(no, nv, aibj)
	Ω2 = copy(aibj)
	@tensoropt aibj[a, i, b, j] += Ω2[b, j, a, i]
	aibj.*=0.5
end

function create_u(no, nv, taibj)
	us = zeros(nv, no, nv, no)
	@tensoropt begin
		us[a, i, b, j] = 2 * taibj[a, i, b, j] - taibj[a, j, b, i]
	end
	return us
end
			
function Ω_tangle!(no, nv, Ω_i, Ω_a, Ω_ai, Ω_aibj, Ω)
	Ω[1:no] = Ω_i
    Ω[(no+1):(no+nv)] = Ω_a
    Ω[(no+nv+1):(no+nv+no*nv)] = vec(Ω_ai)
    Ω[(no+nv+no*nv+1):end] = vec(Ω_aibj)
end

function t_untangle(no, nv, t)
	@assert size(t, 1) == no*no*nv*nv + no*nv
	tai = t[begin:no*nv]
	taibj = t[(no*nv+1):end]
	tai = reshape(tai, nv, no)
	taibj = reshape(taibj, nv, no, nv, no)
	return tai, taibj
end

function parameters_untangle(no, nv, ω)
	@assert size(ω, 1) == (no+nv+no*nv+no*no*nv*nv)
	γo = ω[1:no]
	γv = ω[(no+1):(no+nv)]
	t1s, t2s = t_untangle(no, nv, ω[(no+nv+1):end])
	return γo, γv, t1s, t2s
end

function check_symmetry(no, nv, Ω_aibj, id)
	for a in 1:nv, b in 1:nv, i in 1:no, j in 1:no
		if !isapprox(Ω_aibj[a, i, b, j], Ω_aibj[b, j, a, i], atol = 0.00001)
			println("$id is non-symmetric in index [$a,$i,$b,$j]")
		end
	end
end

function make_Ω_function(nocc, nvir, λ, λdag)
    Ω_i = zeros(Float64, nocc)
    Ω_a = zeros(Float64, nvir)
	Ω_ai = zeros(Float64, nvir, nocc)
	Ω_aibj = zeros(Float64, nvir, nocc, nvir, nocc)
	t1s = zeros(Float64, nvir, nocc)
	t2s = zeros(Float64, nvir, nocc, nvir, nocc)
    γo = zeros(Float64, nocc)
    γv = zeros(Float64, nvir)
    function construct!(Ω_full, ω)
		γo, γv, t1s, t2s = parameters_untangle(nocc, nvir, ω)
        Ω_i!(Ω_i, γo, γv, t1s, t2s, λ, λdag)
        Ω_a!(Ω_a, γo, γv, t1s, t2s, λ, λdag)
		Ω_ai!(Ω_ai, γo, γv, t1s, t2s, λ, λdag)
		Ω_aibj!(Ω_aibj, γo, γv, t1s, t2s, λ, λdag)

		Ω_tangle!(nocc, nvir, Ω_i, Ω_a, Ω_ai, Ω_aibj, Ω_full)
		Ω_full
    end
end

function make_jacobianT_map(no, nv, ω, λ, λdag)
	γo, γv, t1s, t2s = parameters_untangle(no, nv, ω)

	γos = zeros(no)
	γvs = zeros(nv)
	t1ss = zeros(nv, no)
	t2ss = zeros(nv, no, nv, no)

	AT_a = zeros(nv)
	AT_i = zeros(no)
	AT_ai = zeros(nv, no)
	AT_aibj = zeros(nv, no, nv, no)

	LinearMap(length(ω); ismutating = true) do AT_full, ωs
                AT_i.=0; AT_a.=0; AT_ai.=0; AT_aibj.=0
		γos, γvs, t1ss, t2ss = parameters_untangle(no, nv, ωs)
		
		jacobianT_map_i_γo!(AT_i, γo, γv, t1s, t2s, λ, λdag, γos)
		jacobianT_map_a_γo!(AT_i, γo, γv, t1s, t2s, λ, λdag, γvs)
		jacobianT_map_ai_γo!(AT_i, γo, γv, t1s, t2s, λ, λdag, t1ss)
		jacobianT_map_aibj_γo!(AT_i, γo, γv, t1s, t2s, λ, λdag, t2ss)

		jacobianT_map_i_γv!(AT_a, γo, γv, t1s, t2s, λ, λdag, γos)
		jacobianT_map_a_γv!(AT_a, γo, γv, t1s, t2s, λ, λdag, γvs)
		jacobianT_map_ai_γv!(AT_a, γo, γv, t1s, t2s, λ, λdag, t1ss)
		jacobianT_map_aibj_γv!(AT_a, γo, γv, t1s, t2s, λ, λdag, t2ss)

		jacobianT_map_i_t1s!(AT_ai, γo, γv, t1s, t2s, λ, λdag, γos)
		jacobianT_map_a_t1s!(AT_ai, γo, γv, t1s, t2s, λ, λdag, γvs)
		jacobianT_map_ai_t1s!(AT_ai, γo, γv, t1s, t2s, λ, λdag, t1ss)
		jacobianT_map_aibj_t1s!(AT_ai, γo, γv, t1s, t2s, λ, λdag, t2ss)

		jacobianT_map_i_t2s!(AT_aibj, γo, γv, t1s, t2s, λ, λdag, γos)
		jacobianT_map_a_t2s!(AT_aibj, γo, γv, t1s, t2s, λ, λdag, γvs)
		jacobianT_map_ai_t2s!(AT_aibj, γo, γv, t1s, t2s, λ, λdag, t1ss)
		jacobianT_map_aibj_t2s!(AT_aibj, γo, γv, t1s, t2s, λ, λdag, t2ss)
	
		Ω_tangle!(no, nv, AT_i, AT_a, AT_ai, AT_aibj, AT_full)
		AT_full
	end
end

function make_jacobian_map(no, nv, ω, λ, λdag)
	γo, γv, t1s, t2s = parameters_untangle(no, nv, ω)

	γos = zeros(no)
	γvs = zeros(nv)
	t1ss = zeros(nv, no)
	t2ss = zeros(nv, no, nv, no)

	A_a = zeros(nv)
	A_i = zeros(no)
	A_ai = zeros(nv, no)
	A_aibj = zeros(nv, no, nv, no)

	LinearMap(length(ω); ismutating = true) do A_full, ωs
		γos, γvs, t1ss, t2ss = parameters_untangle(no, nv, ωs)
		
		jacobian_map_i_γo!(A_i, γo, γv, t1s, t2s, λ, λdag, γos)
		jacobian_map_i_γv!(A_i, γo, γv, t1s, t2s, λ, λdag, γvs)
		jacobian_map_i_t1s!(A_i, γo, γv, t1s, t2s, λ, λdag, t1ss)
		jacobian_map_i_t2s!(A_i, γo, γv, t1s, t2s, λ, λdag, t2ss)

		jacobian_map_a_γo!(A_a, γo, γv, t1s, t2s, λ, λdag, γos)
		jacobian_map_a_γv!(A_a, γo, γv, t1s, t2s, λ, λdag, γvs)
		jacobian_map_a_t1s!(A_a, γo, γv, t1s, t2s, λ, λdag, t1ss)
		jacobian_map_a_t2s!(A_a, γo, γv, t1s, t2s, λ, λdag, t2ss)

		jacobian_map_ai_γo!(A_ai, γo, γv, t1s, t2s, λ, λdag, γos)
		jacobian_map_ai_γv!(A_ai, γo, γv, t1s, t2s, λ, λdag, γvs)
		jacobian_map_ai_t1s!(A_ai, γo, γv, t1s, t2s, λ, λdag, t1ss)
		jacobian_map_ai_t2s!(A_ai, γo, γv, t1s, t2s, λ, λdag, t2ss)

		jacobian_map_aibj_γo!(A_aibj, γo, γv, t1s, t2s, λ, λdag, γos)
		jacobian_map_aibj_γv!(A_aibj, γo, γv, t1s, t2s, λ, λdag, γvs)
		jacobian_map_aibj_t1s!(A_aibj, γo, γv, t1s, t2s, λ, λdag, t1ss)
		jacobian_map_aibj_t2s!(A_aibj, γo, γv, t1s, t2s, λ, λdag, t2ss)
	
		Ω_tangle!(no, nv, A_i, A_a, A_ai, A_aibj, A_full)
		A_full
	end
end

function make_A_left_trans(no, nv, λ, λdag)
	function construct_AT(ω)
		make_jacobianT_map(no, nv, ω, λ, λdag)
	end
end

function make_A_right_trans(no, nv, λ, λdag)
	function construct_A(ω)
		make_jacobian_map(no, nv, ω, λ, λdag)
	end
end

function make_eta_function(no, nv, λ, λdag)
	#η = zeros(no+nv+no*nv+no*nv*no*nv)
	η_i = zeros(no)
	η_a = zeros(nv)
	η_ai = zeros(nv,no)
	η_aibj = zeros(nv,no,nv,no)
	function eta(η, ω)
		γo, γv, t1s, t2s = parameters_untangle(no, nv, ω)
		#η_i, η_a, η_ai, η_aibj = parameters_untangle(no, nv, η)
		
		eta_func_a!(no, nv, γo, γv, t1s, t2s, λ, λdag, η_a)
		eta_func_i!(no, nv, γo, γv, t1s, t2s, λ, λdag, η_i)
		eta_func_ai!(no, nv, γo, γv, t1s, t2s, λ, λdag, η_ai)
		eta_func_aibj!(no, nv, γo, γv, t1s, t2s, λ, λdag, η_aibj)
		Ω_tangle!(no, nv, η_i, η_a, η_ai, η_aibj, η)
		η
	end
end

function construct_g_mo(mol, hf)
    nao = py"int"(mol.nao)
    nocc = mol.nelectron ÷ 2
    nvir = nao - nocc

    occ = 1:nocc
    vir = (nocc+1):nao

    mo_coeff = hf.mo_coeff
	ao2mo = pyscf.ao2mo
	mo_coeff = hf.mo_coeff

	eri_ao = mol.intor("int2e")

	eri_mo = ao2mo.full(hf._eri, mo_coeff)
	eri_mo_full = ao2mo.restore(1, eri_mo, size(mo_coeff,2))
	w_oooo = eri_mo_full[occ, occ, occ, occ]
	w_ooov = eri_mo_full[occ, occ, occ, vir]
	w_oovo = eri_mo_full[occ, occ, vir, occ]
	w_oovv = eri_mo_full[occ, occ, vir, vir]
	w_ovoo = eri_mo_full[occ, vir, occ, occ]
	w_ovov = eri_mo_full[occ, vir, occ, vir]
	w_ovvo = eri_mo_full[occ, vir, vir, occ]
	w_ovvv = eri_mo_full[occ, vir, vir, vir]
	w_vooo = eri_mo_full[vir, occ, occ, occ]
	w_voov = eri_mo_full[vir, occ, occ, vir]
	w_vovo = eri_mo_full[vir, occ, vir, occ]
	w_vovv = eri_mo_full[vir, occ, vir, vir]
	w_vvoo = eri_mo_full[vir, vir, occ, occ]
	w_vvov = eri_mo_full[vir, vir, occ, vir]
	w_vvvo = eri_mo_full[vir, vir, vir, occ]
	w_vvvv = eri_mo_full[vir, vir, vir, vir]

	w_oooo, w_ooov, w_oovo, w_ovoo, w_vooo, w_oovv, w_ovov, w_ovvo, w_voov, w_vovo, w_vvoo, w_ovvv, w_vovv, w_vvov, w_vvvo, w_vvvv
end

function construct_fock(mol, hf)
    nao = py"int"(mol.nao)
	nocc = mol.nelectron ÷ 2
	nvir = nao - nocc
	
	F_ao = hf.get_fock()
	C = hf.mo_coeff
	F_mo = C' * F_ao * C
	
	nocc = Int(mol.nelectron ÷ 2)
	nmo = size(F_mo, 1)
	F_oo = F_mo[1:nocc, 1:nocc]
	F_ov = F_mo[1:nocc, nocc+1:end]
	F_vo = F_mo[nocc+1:end, 1:nocc]
	F_vv = F_mo[nocc+1:end, nocc+1:end]
	F_oo, F_ov, F_vo, F_vv
end

function create_Ls(g_oooo, g_ooov, g_oovo, g_ovoo, g_vooo, g_oovv, g_ovov, g_ovvo, g_voov, g_vovo, g_vvoo, g_ovvv, g_vovv, g_vvov, g_vvvo, g_vvvv)
	global L_oooo = similar(g_oooo)
	global L_ooov = similar(g_ooov)
	global L_oovo = similar(g_oovo)
	global L_ovoo = similar(g_ovoo)
	global L_vooo = similar(g_vooo)
	global L_oovv = similar(g_oovv)
	global L_ovov = similar(g_ovov)
	global L_ovvo = similar(g_ovvo)
	global L_voov = similar(g_voov)
	global L_vovo = similar(g_vovo)
	global L_vvoo = similar(g_vvoo)
	global L_ovvv = similar(g_ovvv)
	global L_vovv = similar(g_vovv)
	global L_vvov = similar(g_vvov)
	global L_vvvo = similar(g_vvvo)
	global L_vvvv = similar(g_vvvv)
	@tensor begin
		L_oooo[i, j, k, l] = 2 * g_oooo[i, j, k, l] - g_oooo[i, l, k, j]
		L_ooov[i, j, k, a] = 2 * g_ooov[i, j, k, a] - g_ovoo[i, a, k, j]
		L_oovo[i, j, a, k] = 2 * g_oovo[i, j, a, k] - g_oovo[i, k, a, j]
		L_ovoo[i, a, j, k] = 2 * g_ovoo[i, a, j, k] - g_ooov[i, k, j, a]
		L_vooo[a, i, j, k] = 2 * g_vooo[a, i, j, k] - g_vooo[a, k, j, i]
		L_oovv[i, j, a, b] = 2 * g_oovv[i, j, a, b] - g_ovvo[i, b, a, j]
		L_ovov[i, a, j, b] = 2 * g_ovov[i, a, j, b] - g_ovov[i, b, j, a]
		L_ovvo[i, a, b, j] = 2 * g_ovvo[i, a, b, j] - g_oovv[i, j, b, a]
		L_voov[a, i, j, b] = 2 * g_voov[a, i, j, b] - g_vvoo[a, b, j, i]
		L_vovo[a, i, b, j] = 2 * g_vovo[a, i, b, j] - g_vovo[a, j, b, i]
		L_vvoo[a, b, i, j] = 2 * g_vvoo[a, b, i, j] - g_voov[a, j, i, b]
		L_ovvv[i, a, b, c] = 2 * g_ovvv[i, a, b, c] - g_ovvv[i, c, b, a]
		L_vovv[a, i, b, c] = 2 * g_vovv[a, i, b, c] - g_vvvo[a, c, b, i]
		L_vvov[a, b, i, c] = 2 * g_vvov[a, b, i, c] - g_vvov[a, c, i, b]
		L_vvvo[a, b, c, i] = 2 * g_vvvo[a, b, c, i] - g_vovv[a, i, c, b]
		L_vvvv[a, b, c, d] = 2 * g_vvvv[a, b, c, d] - g_vvvv[a, d, c, b]
	end
end

function density_1e(no, nv, ω, ωbar)
	nmo = no + nv
	D = zeros(nmo, nmo)
	
	γobar, γvbar, t1sbar, t2sbar = parameters_untangle(no, nv, ωbar)
	γo, γv, t1s, t2s = parameters_untangle(no, nv, ω)

	o = 1:no
	v = (no+1):nmo
	
	δ_oo = Matrix(I, no, no)
	δ_vv = Matrix(I, nv, nv)
	D_oo = @view D[o,o]
	D_vv = @view D[v,v]
	D_ov = @view D[o,v]
	D_vo = @view D[v,o]

	@tensoropt begin
		D_oo[i,j] = 2 * δ_oo[i,j]
		D_oo[i,j] -= γo[i] * γobar[j]
		D_oo[i,j] -= t1s[a,i] * t1sbar[a,j]
		D_oo[i,j] += t1sbar[a,j] * γo[i] * γv[a]
		D_oo[i,j] -= t2s[a,i,b,k] * t2sbar[a,j,b,k]
		D_oo[i,j] -= t2s[a,i,b,k] * t2sbar[b,k,a,j]

		D_ov[i,a] = 2 * t1s[a,i]
		D_ov[i,a] -= t1s[a,j] * γo[i] * γobar[j]
		D_ov[i,a] -= t1s[b,i] * γv[a] * γvbar[b]
		D_ov[i,a] += 2 * t2s[a,i,b,j] * t1sbar[b,j]
		D_ov[i,a] -= t2s[a,j,b,i] * t1sbar[b,j]
		D_ov[i,a] -= t1s[a,j] * t1s[b,i] * t1sbar[b,j]
		D_ov[i,a] += t1s[a,j] * t1sbar[b,j] * γo[i] * γv[b]
		D_ov[i,a] -= t1s[b,i] * t1sbar[b,j] * γv[a] * γo[j]
		D_ov[i,a] -= t1s[a,j] * t2s[b,i,c,k] * t2sbar[b,j,c,k]
		D_ov[i,a] -= t1s[a,j] * t2s[b,i,c,k] * t2sbar[c,k,b,j]
		D_ov[i,a] -= t1s[b,i] * t2s[a,j,c,k] * t2sbar[b,j,c,k]
		D_ov[i,a] -= t1s[b,i] * t2s[a,j,c,k] * t2sbar[c,k,b,j]
		D_ov[i,a] += t2s[a,j,b,k] * t2sbar[b,k,c,j] * γo[i] * γv[c]
		D_ov[i,a] += t2s[a,j,b,k] * t2sbar[c,j,b,k] * γo[i] * γv[c]
		D_ov[i,a] -= t2s[b,i,c,j] * t2sbar[b,k,c,j] * γv[a] * γo[k]
		D_ov[i,a] -= t2s[b,i,c,j] * t2sbar[c,j,b,k] * γv[a] * γo[k]
		
		D_vo[a,i] = t1sbar[a,i]

		D_vv[a,b] = γv[b] * γvbar[a]
		D_vv[a,b] += t1s[b,i] * t1sbar[a,i]
		D_vv[a,b] += t1sbar[a,i] * γv[b] * γo[i]
		D_vv[a,b] += t2s[b,i,c,j] * t2sbar[a,i,c,j]
		D_vv[a,b] += t2s[b,i,c,j] * t2sbar[c,j,a,i]
	end

	println("Regular γo: ", γo)
	println("γo_bar: ", γobar)
	println("\nRegular γv: ", γv)
	println("γv_bar: ", γvbar)
	for p in o
		if D[p,p] > 2
			D_γo = 0
			D_γo2= 0
			D_t1 = 0
			D_t2 = 0
			D_γo -= γobar[p] * γo[p]
			for a in 1:nv, j in o
				D_γo2 -= δ_oo[p,j] * t1sbar[a,p] * γo[j] * γv[a]
				D_t1 -= δ_oo[j,p] * t1sbar[a,j] * t1s[a,p]
				for b in 1:nv, k in o
					D_t2 -= δ_oo[j,p] * t2sbar[a,j,b,k] * t2s[a,p,b,k]
					D_t1 -= δ_oo[j,p] * t2sbar[a,k,b,j] * t2s[a,k,b,p]
				end
			end
			println("Overfilled occupied nr. ", p,": ", D[p,p])
			println("Contributions:")
			println("γo: ", D_γo)
			println("γo2: ", D_γo2)
			println("t1s: ", D_t1)
			println("t2s: ", D_t2)
		end
	end
	for q in v
		if D[q,q] < 0
			D_γv = 0
			D_γv2 = 0
			D_t1 = 0
			D_t2 = 0
			for a in 1:nv, b in 1:nv
				D_γv += δ_vv[a,b] * γvbar[a] * γv[b]
				for i in o
					D_t1 += δ_vv[a,b] * t1sbar[a,i] * t1s[b,i]
					D_γv2 += δ_vv[a,b] * t1sbar[a,i] * γv[b] * γo[i]
					for c in 1:nv, j in o
						D_t2 += δ_vv[a,b] * t2sbar[a,i,c,j] * t2s[b,i,c,j]
						D_t2 += δ_vv[a,b] * t2sbar[c,i,a,j] * t2s[b,j,c,i]
					end
				end
			end
			println("Overremoved virtual nr. ", q-no, ": ", D[q,q])
			println("Contributions:")
			println("γv: ", D_γv)
			println("γv2: ", D_γv2)
			println("t1s: ", D_t1)
			println("t2s: ", D_t2)
		end
	end
	D
end

function compare_CCSD(mol, pb_corr)
	ccsd = mol.CCSD().run()
	dm1 = ccsd.make_rdm1()
	#print(diag(dm1))		#bugtest line
	println("E_ccsd-E_pbccsd = ", ccsd.e_corr-pb_corr)
end

function run_multiple_lambdas(nocc, nvir, mol, hf, λ_strengths)
	e_list = zeros(size(λ_strengths, 1))
	e_list.=NaN
	n_list = zeros(size(λ_strengths, 1))
	n_list.=NaN
	j = 1
	for i in λ_strengths
		λ[1:2].=-i
                λdag[3:4].=-i
		println("Using λs: ", λ)
		println("Using λdags: ", λdag)

		Ω_func = make_Ω_function(nocc, nvir, λ, λdag)
    	A_func = make_A_right_trans(nocc, nvir, λ, λdag)    # This uses LinearMaps for Jacobian
    	ω = @time solve_ω(Ω_func, A_func, size(F_oo, 1), size(F_vv, 1), λ, λdag, 1e-10)
    	n_ij = (nocc * (nocc + 1)) ÷ 2

    	e_func = make_pbcc_E_func(nocc, nvir, λ, λdag)
    	e_hf, e_pb, e_cc1, e_cc2 = e_func(ω)
	e_corr = e_pb + e_cc1 + e_cc2
    	η_func = make_eta_function(nocc, nvir, λ, λdag)
    	η = zeros(size(ω))
    	η = η_func(η, ω)

		AT_func = make_A_left_trans(nocc, nvir, λ, λdag)
		AT = AT_func(ω)
    	ωbar = idrs(AT, -η; verbose=false, abstol=1e-6, s=1000)
		
    	D, D_diag = density_1e(nocc, nvir, ω, ωbar)
		println(D_diag)

    	N_el = tr(D)
    	@show N_el

    	ΔN_el = N_el - mol.nelectron
    	@show ΔN_el
		if abs(e_corr) < 1
			e_list[j] = e_corr
			n_list[j] = ΔN_el
    	else
			e_list[j] = NaN 
			n_list[j] = NaN 
		end
		println("------------ end of run nr. ", j, "/", size(λ_strengths, 1),"------------")
		savefig(plot(λ_strengths, e_list), "Energies.png")
		savefig(plot(λ_strengths, n_list), "Density.png")
		j += 1
	end
end



function solve_ω(Ω_func, A_func, no, nv, λ, λdag, tol = 1e-6)
	println("Entered function")
	ω = zeros(Float64, (no+nv+no*nv+no*nv*no*nv))
ω[1:2].=-0.03345285130136979
ω[3:4].=0.029220840240760076
ω[5]=8.824673291337335e-5
ω[5+3]=8.824673291337335e-5
ω[(2+2+2*2+1)]=-0.11534048774597826
ω[end]=-0.11534048774597826
	ω_test = copy(ω)
	dω = copy(ω)
	Ω = copy(ω)
	Ω_func(Ω, ω)
	res = maximum(abs, Ω)
	res_test = copy(res)
	norm_Ω = norm(Ω)

	step_size = 1
	#iter_method = idrs!
	iter_method = gmres!
	i = 1
	while res > tol && i < 80 && res < 1e8
		println("i = $i, |Ω| = ", res)
                println("|Ω_i| = ", maximum(abs, Ω[1:no]))
                println("|Ω_a| = ", maximum(abs, Ω[(no+1):(no+nv)]))
                println("|Ω_ai| = ", maximum(abs, Ω[(no+nv+1):(no+nv+no*nv)]))
                println("|Ω_aibj| = ", maximum(abs, Ω[(no+nv+no*nv+1):end]))
                
		
                #Ω[(no+nv+1):(no+nv+no*nv)].=0
		#ω[(no+nv+1):(no+nv+no*nv)].=0
		A = A_func(ω)
		iter_method(dω, A, Ω, verbose = true, abstol=max(res, tol) * 1e-2)		#Calculate using iteration method
		
		axpy!(-step_size, dω, ω)
		
		Ω_func(Ω, ω)
                #Ω[(no+nv+1):(no+nv+no*nv)].=0
		#ω[(no+nv+1):(no+nv+no*nv)].=0
		res = maximum(abs, Ω)
		new_norm = norm(Ω)
		dω .*= new_norm/norm_Ω
		norm_Ω = new_norm
		i += 1
	end

	println("i = $i, |Ω| = ", res)
	println("γo = ", ω[1:no])
	println("γv = ", ω[(no+1):(no+nv)])
	if i > 98 || norm_Ω > 1e6
		return ω.+Inf
	end
	ω
end

function output_line(output_file, output_line)
	if output_file != 0
		open(output_file, "a") do io
			write(io, "$output_line\n")
		end
	else println("No output file")
	end
end

function get_AOs(no, nv, mol, hf)
	C = hf.mo_coeff
	cubegen.orbital(mol, "mo0.cube", C[1, :])  # MO 0
	cubegen.orbital(mol, "mo1.cube", C[2, :])  # MO 1
end

function solve_pbcc(mol, λ, λdag, output_file = 0, n_eig=0)
    hf = mol.HF().run()
    nao = py"int"(mol.nao)
    
    nocc = mol.nelectron ÷ 2
    nvir = nao - nocc
C = hf.mo_coeff
	
	#get_AOs(nocc, nvir, mol, hf)

	#println("no: ", nocc, ", nv: ", nvir, "nao_A: ", naoA)
    global F_oo, F_ov, F_vo, F_vv = construct_fock(mol, hf)
println(diag(F_oo))
println(diag(F_vv))

    global g_oooo, g_ooov, g_oovo, g_ovoo, g_vooo, g_oovv, g_ovov, g_ovvo, g_voov, g_vovo, g_vvoo, g_ovvv, g_vovv, g_vvov, g_vvvo, g_vvvv = @time construct_g_mo(mol, hf)
	create_Ls(g_oooo, g_ooov, g_oovo, g_ovoo, g_vooo, g_oovv, g_ovov, g_ovvo, g_voov, g_vovo, g_vvoo, g_ovvv, g_vovv, g_vvov, g_vvvo, g_vvvv)

	λ_strengths = [1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 1e-2, 5e-2]
    λdag[1:nocc].=0.1
    λ[(nocc+1):(nocc+nvir)].=0.1

	#run_multiple_lambdas(nocc, nvir, mol, hf, λ_strengths)
	output_line(output_file, "λs set to $λ\nλdags set to $λdag\n")

	Ω_func = make_Ω_function(nocc, nvir, λ, λdag)
    e_func = make_pbcc_E_func(nocc, nvir, λ, λdag)
    A_func = make_A_right_trans(nocc, nvir, λ, λdag)    # This uses LinearMaps for Jacobian
println("λs: ", λ)
println("λdags: ", λdag)

    ω = @time solve_ω(Ω_func, A_func, size(F_oo, 1), size(F_vv, 1), λ, λdag, 1e-8)

	γo_final, γv_final, t1s_final, t2s_final = parameters_untangle(nocc, nvir, ω)

	savefig(heatmap(reshape(ω, :, 1)), "om_final.png")
    n_ij = (nocc * (nocc + 1)) ÷ 2

    e_func = make_pbcc_E_func(nocc, nvir, λ, λdag)
	e_hf, e_pb, e_cc1, e_cc2 = e_func(ω)
	e_corr = e_pb+e_cc1+e_cc2

    @show e_corr
    e_tot = e_corr + e_hf + hf.energy_nuc()
    @show e_tot
	compare_CCSD(mol, e_corr)
	println()
	output_line(output_file, "Converged energies: ")
	output_line(output_file, "Total Particle-Breaking contribution = $e_pb")
	output_line(output_file, "Total Single-excitation contribution = $e_cc1")
	output_line(output_file, "Total Double-excitation contribution = $e_cc2")
	output_line(output_file, "Total correlation energy = $e_corr")
	output_line(output_file, "Total PBCCSD energy = $e_tot\n")

	
    η_func = make_eta_function(nocc, nvir, λ, λdag)
    η = zeros(size(ω))
    η = η_func(η, ω)

    tol = 1e-10
    AT_func = make_A_left_trans(nocc, nvir, λ, λdag)
	AT = AT_func(ω)
	ωbar = zeros(size( ω,1))
    gmres!(ωbar, AT, -η; verbose=false, abstol=tol, restart = 500)
	
    Ω = zeros(size(ω))
    Ω_func(Ω, ω)

    Δe_Λ = Ω ⋅ ωbar
    e_Λ = e_tot + Δe_Λ

    @show Δe_Λ e_Λ
	
    D = density_1e(nocc, nvir, ω, ωbar)

	println(diag(D))
    N_el = tr(D)
    @show N_el

    ΔN_el = N_el - mol.nelectron
    @show ΔN_el

	output_line(output_file, "N_el: $N_el")
	output_line(output_file, "ΔN_el: $ΔN_el")
	output_line(output_file, "Occupied orbitals:")
	output_line(output_file, diag(D)[1:nocc])
	output_line(output_file, "Virtual orbitals:")
	output_line(output_file, diag(D)[(nocc+1):(nocc+nvir)])

	C = hf.mo_coeff
	D_ao = C * D * C'
	cubegen.density(mol, "total_density.cube", D_ao, nx = 100, ny = 100, nz = 100)

    if n_eig > 0
        A = make_jacobian_right_transformation(F_oo, F_vv, g_oooo, g_vvvv, g_ovov, γ, λ_oo, λ_vv)
        vals_R, vecs_R = eigsolve(A, length(γ), n_eig, :SR; verbosity=2, tol=1e-5, eager=true)

        println("Eigenvalues:")
        display(vals_R)
        println()

        for i in 1:n_eig
            println("\ni: ", i)

            println("E: ", vals_R[i])
            curvec = vecs_R[i]
            println("\nDominant R_ij:")
            #print_dominant_xy(curvec, nocc)
            println("\nDominant R_ab:")
            #print_dominant_xy((@view curvec[begin+n_ij:end]), nvir)
            println()
        end

        noise = randn(length(γ))
        noise .*= 1e-3 / norm(noise)

        vals_L, vecs_L = eigsolve(AT, vecs_R[1] .* noise, n_eig, :SR; verbosity=2, tol=1e-5, eager=true)

        println("Eigenvalues:")
        display(vals_L)
        display(abs.(vals_L[1:n_eig] .- vals_R[1:n_eig]))
        println()

        for i in 1:n_eig
            println("\ni: ", i)

            println("E: ", vals_L[i])
            curvec = vecs_L[i]

            ovlp = curvec ⋅ vecs_R[i]
            @show ovlp
            curvec .*= 1 / ovlp

            println("\nDominant L_ij:")
            print_dominant_xy(curvec, nocc)
            println("\nDominant L_ab:")
            print_dominant_xy((@view curvec[begin+n_ij:end]), nvir)
            println()
        end

        for i in 1:n_eig
            println("\ni: ", i)

            R0 = η ⋅ vecs_R[i] / (vals_R[i] - e_Λ)

            D_ex = density_ex_1e_from_packed(vecs_L[i], R0, vecs_R[i], γ, nocc, nvir)

            N_el_ex = tr(D_ex)
            @show N_el_ex

            ΔN_el_ex = N_el_ex - N_el
            @show ΔN_el_ex
        end
    end

    ω
end

function make_gaussian_λ(mol, λ0, r; do_occ=true, do_vir=true)
    hf = mol.HF().run()

    nao = py"int"(mol.nao)
    nocc = mol.nelectron ÷ 2
    nvir = nao - nocc

    λ = zeros(nocc)
    λdag = zeros(nvir)

    if do_occ
        for i in 1:nocc
            e = hf.mo_energy[i]
            λ[i] = λ0 * exp(-r * e^2)
        end
    end

    if do_vir
        for a in 1:nvir
            e = hf.mo_energy[nocc+a]
            λdag[a] = λ0 * exp(-r * e^2)
        end
    end

    λ, λdag
end

function make_binary_λ(mol, strength, do_occ = true, do_vir = true)
    nao = py"int"(mol.nao)
    nocc = mol.nelectron ÷ 2
    nvir = nao - nocc
	λ = zeros(nocc+nvir)
	λdag = zeros(nvir+nocc)
	if do_occ == true
		λ[nocc] = strength
                λdag[nocc] = -strength
	end
	if do_vir == true
                λ[(nocc+1)] = strength
		λdag[(nocc+1)] = -strength
	end
	λ, λdag
end

function run_multiple_geometries(folder, basis, strength, spread, output_file = "output.out", do_const = true)
	λ = 0
	λdag = 0
	for file in filter(f -> endswith(f, ".xyz"), readdir(folder, join=true))
		data = readlines(file)
		geometry = pyscf.gto.Mole()
		geometry.atom = data[3:end]
		geometry.basis = basis
		geometry.build()
		if spread == Inf
			λ, λdag = make_binary_λ(geometry, strength)
		else
			λ, λdag = make_gaussian_λ(geometry, strength, spread)
		end
	end
	for file in filter(f -> endswith(f, ".xyz"), readdir(folder, join=true))
		
		data = readlines(file)

		println("Processing: $file")
		open(output_file, "a") do io
			write(io, "----------$file----------\n")
			for line in data
				write(io, line)
				write(io,"\n")
			end
		end
		println(data[2])
		
		geometry = pyscf.gto.Mole()
		geometry.atom = data[3:end]
		geometry.basis = basis
		geometry.build()

		if do_const == false
			if spread == Inf
				λ, λdag = make_binary_λ(geometry, strength)
			else
				λ, λdag = make_gaussian_λ(geometry, strength, spread)
			end
		end
		solve_pbcc(geometry, λ, λdag, output_file)
	end
end

function find_λ(mol, sol)
    mol_hf = mol.HF().run()
	sol_hf = sol.HF().run()

    mol_nao = py"int"(mol.nao)
    mol_nocc = mol.nelectron ÷ 2
    mol_nvir = mol_nao - mol_nocc

    λ = zeros(mol_nocc)
    λdag = zeros(mol_nvir)
	
    sol_nao = py"int"(sol.nao)
    sol_nocc = sol.nelectron ÷ 2
    sol_nvir = sol_nao - sol_nocc

	mol_C = mol_hf.mo_coeff
	sol_C = sol_hf.mo_coeff

	F_mol_ao = mol_hf.get_fock()
	F_sol_ao = sol_hf.get_fock()

	g_ao = sol.intor("int2e")
	h_ao = sol.intor("int1e_ovlp")

	for A in (mol_nao+sol_nocc+1):sol_nao, B in (mol_nao+sol_nocc+1):sol_nao, I in (mol_nao+1):(mol_nao+sol_nocc)
		for i in 1:mol_nocc
			λ[i] += g_ao[A, i, B, I]+g_ao[A, I, B, i]
		end
	end
	for A in (mol_nao+sol_nocc+1):sol_nao, I in (mol_nao+1):(mol_nao+sol_nocc), J in (mol_nao+1):(mol_nao+sol_nocc)
		for a in 1:mol_nvir
			λdag[a] += g_ao[A, I, (a+mol_nocc), J] - g_ao[(a+mol_nocc), I, A, J]
		end
	end
	for A in (mol_nao+sol_nocc+1):sol_nao
		for i in 1:mol_nocc
			λ[i] += h_ao[A,i]
		end
	end
	for I in (mol_nao+1):(mol_nao+sol_nocc)
		for a in 1:mol_nvir
			λdag[a] += h_ao[(a+mol_nocc),I]
		end
	end

	λ, λdag
end

function swap_matrix(mol, C, nA)
    S = mol.intor("int1e_ovlp")
    n = size(S, 1)
    @assert n == 2nA "Expected two fragments with $nA AOs each"
    Π = zeros(n, n)
    for μ in 1:nA
        Π[μ + nA, μ] = 1
        Π[μ, μ + nA] = 1
    end
    C' * S * Π * C
end

function swap_ω(ω, Ro, Rv, no, nv)
    γo, γv, t1, t2 = parameters_untangle(no, nv, ω)
    @tensor t1r[a, i] := Rv[a, b] * Ro[i, j] * t1[b, j]
    @tensor t2r[a, i, b, j] := Rv[a, c] * Ro[i, k] * Rv[b, d] * Ro[j, l] * t2[c, k, d, l]
    [Ro * γo; Rv * γv; vec(t1r); vec(t2r)]
end

function check_swap_symmetry(mol, hf, nocc, ω, λ, λdag, Ω_func, nA)
    C = hf.mo_coeff
    R = swap_matrix(mol, C, nA)
    o = 1:nocc
    v = nocc+1:size(R, 1)
    Ro, Rv = R[o, o], R[v, v]
    nvir = length(v)

    println("R orthogonal:   ", maximum(abs, R' * R - I))
    println("R mixes o/v:    ", maximum(abs, R[o, v]))

    F = C' * hf.get_fock() * C
    println("Fock:           ", maximum(abs, R' * F * R - F))

    eri = pyscf.ao2mo.restore(1, pyscf.ao2mo.full(mol, C), size(C, 2))
    @tensoropt eri_s[p, q, r, s] := R[a, p] * R[b, q] * R[c, r] * R[d, s] * eri[a, b, c, d]
    println("Integrals:      ", maximum(abs, eri_s - eri))

    println("λ:              ", maximum(abs, R * λ - λ))
    println("λdag:           ", maximum(abs, R * λdag - λdag))

    x = 0.01 * randn(length(ω))
    Ωa = zeros(length(ω))
    Ωb = zeros(length(ω))
    Ω_func(Ωa, swap_ω(x, Ro, Rv, nocc, nvir))
    Ω_func(Ωb, x)
    println("Ω equivariance: ", maximum(abs, Ωa - swap_ω(Ωb, Ro, Rv, nocc, nvir)))

    Ωs = zeros(length(ω))
    Ω_func(Ωs, ω)
    println("Residual at ω:  ", maximum(abs, Ωs))
    println("Solution ω:     ", maximum(abs, swap_ω(ω, Ro, Rv, nocc, nvir) - ω))
    Ro, Rv
end

H2O = pyscf.gto.Mole()
H2O.atom = "
H      5.180957    1002.090311   -0.376197
O      4.720088    1002.149309   -1.219625
H      3.789240    1002.113041   -0.975796
"

H2O_2 = pyscf.gto.Mole()
H2O_2.atom = "
N     -1003.731512   -0.230146   -2.258078
H     -1003.415694    0.482764   -1.613451
H     -1002.902397   -0.623504   -2.683955
H     -1004.236325    0.247210   -2.993476
"

H2O2 = pyscf.gto.Mole()
H2O2.atom = "
H      5.180957    1002.090311   -0.376197
O      4.720088    1002.149309   -1.219625
H      3.789240    1002.113041   -0.975796
N     -1003.731512   -0.230146   -2.258078
H     -1003.415694    0.482764   -1.613451
H     -1002.902397   -0.623504   -2.683955
H     -1004.236325    0.247210   -2.993476
"
H2O.basis = "cc-pvdz"
H2O2.basis = "cc-pvdz"
H2O_2.basis = "cc-pvdz"
H2O_2.build()
H2O2.build()
H2O.build()


λ, λdag = make_binary_λ(H2O2, 0.0000)
λ.= 0.0
λdag.= 0.0
ω_start = solve_pbcc(H2O2, λ, λdag, "AOdensity2H2O.out")

