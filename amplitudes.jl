function Ω_i!(Ω_i, γo, γv, t1s, t2s, λ, λdag)
   no = size(γo, 1)
   nv = size(γv, 1)
   λo = @view λ[1:no]
   λv = @view λ[(no+1):(no+nv)]
   λdago = @view λdag[1:no]
   λdagv = @view λdag[(1+no):(no+nv)]

   @tensoropt begin
       Ω_i[i]  = λo[i]
       Ω_i[i] -= γo[j] * F_oo[j,i]
       Ω_i[i] += λv[a] * t1s[a,i]
       Ω_i[i] += 2 * γo[i] * γv[a] * λv[a]
       Ω_i[i] -= 2 * γo[i] * γo[j] * λdago[j]
       Ω_i[i] -= γo[j] * F_ov[j,a] * t1s[a,i]
       Ω_i[i] -= γo[j] * t1s[a,k] * L_ooov[j,i,k,a]
       Ω_i[i] -= γo[j] * L_ovov[j,a,k,b] * t2s[a,i,b,k]
       Ω_i[i] -= γo[j] * t1s[a,i] * t1s[b,k] * L_ovov[j,a,k,b]
   end

end

function Ω_a!(Ω_a, γo, γv, t1s, t2s, λ, λdag)
   no = size(γo, 1)
   nv = size(γv, 1)
   λo = @view λ[1:no]
   λv = @view λ[(no+1):(no+nv)]
   λdago = @view λdag[1:no]
   λdagv = @view λdag[(1+no):(no+nv)]

   @tensoropt begin
      Ω_a[a]  = - λdagv[a]
      Ω_a[a] += γv[b] * F_vv[a,b]
      Ω_a[a] += λdago[i] * t1s[a,i]
      Ω_a[a] += 2 * γv[a] * γv[b] * λv[b]
      Ω_a[a] -= 2 * γv[a] * γo[i] * λdago[i]
      Ω_a[a] -= γv[b] * F_ov[i,b] * t1s[a,i]
      Ω_a[a] += γv[b] * t1s[c,i] * L_vvov[a,b,i,c]
      Ω_a[a] -= γv[b] * L_ovov[i,b,j,c] * t2s[a,i,c,j]
      Ω_a[a] -= γv[b] * t1s[a,i] * t1s[c,j] * L_ovov[i,b,j,c]
   end
end

function Ω_ai!(Ω_ai, γo, γv, t1s, t2s, λ, λdag)
 
   no = size(γo, 1)
   nv = size(γv, 1)
   u = create_u(no, nv, t2s)
    @tensoropt begin
    Ω_ai[a,i]  = F_vo[a,i]
    Ω_ai[a,i] += F_vv[a,b] * t1s[b,i]
    Ω_ai[a,i] -= F_oo[j,i] * t1s[a,j]
    Ω_ai[a,i] += F_ov[j,b] * u[a,i,b,j]
    Ω_ai[a,i] += t1s[b,j] * L_voov[a,i,j,b]
    Ω_ai[a,i] += L_vvov[a,b,j,c] * t2s[b,i,c,j]
    Ω_ai[a,i] -= L_ooov[j,i,k,b] * t2s[a,j,b,k]
    Ω_ai[a,i] += γv[a] * γo[j] * F_oo[j,i]
    Ω_ai[a,i] += γo[i] * γv[b] * F_vv[a,b]
    Ω_ai[a,i] -= F_ov[j,b] * t1s[a,j] * t1s[b,i]
    Ω_ai[a,i] -= t1s[a,j] * t1s[b,k] * L_ooov[j,i,k,b]
    Ω_ai[a,i] += t1s[b,i] * t1s[c,j] * L_vvov[a,b,j,c]
    Ω_ai[a,i] -= t1s[a,j] * L_ovov[j,b,k,c] * t2s[b,i,c,k]
    Ω_ai[a,i] -= t1s[b,i] * L_ovov[j,b,k,c] * t2s[a,j,c,k]
    Ω_ai[a,i] += t1s[b,j] * L_ovov[j,b,k,c] * u[a,i,c,k]
    Ω_ai[a,i] += γv[a] * γo[j] * F_ov[j,b] * t1s[b,i]
    Ω_ai[a,i] -= γo[i] * γv[b] * F_ov[j,b] * t1s[a,j]
    Ω_ai[a,i] += γv[a] * γo[j] * t1s[b,k] * L_ooov[j,i,k,b]
    Ω_ai[a,i] += γo[i] * γv[b] * t1s[c,j] * L_vvov[a,b,j,c]
    Ω_ai[a,i] += γv[a] * γo[j] * L_ovov[j,b,k,c] * t2s[b,i,c,k]
    Ω_ai[a,i] -= γo[i] * γv[b] * L_ovov[j,b,k,c] * t2s[a,j,c,k]
    Ω_ai[a,i] -= t1s[a,j] * t1s[b,i] * t1s[c,k] * L_ovov[j,b,k,c]
    Ω_ai[a,i] += γv[a] * γo[j] * t1s[b,i] * t1s[c,k] * L_ovov[j,b,k,c]
    Ω_ai[a,i] -= γo[i] * γv[b] * t1s[a,j] * t1s[c,k] * L_ovov[j,b,k,c]
   end

end

function Ω_aibj!(Ω_aibj, γo, γv, t1s, t2s, λ, λdag)
   
   no = size(γo, 1)
   nv = size(γv, 1)
   u = create_u(no, nv, t2s)
   @tensoropt begin
    Ω_aibj[a, i, b, j]  = L_vovo[a,i,b,j]
    Ω_aibj[a, i, b, j] += F_vv[a,c] * u[b,j,c,i]
    Ω_aibj[a, i, b, j] += F_vv[b,c] * u[a,i,c,j]
    Ω_aibj[a, i, b, j] -= F_oo[k,i] * u[a,k,b,j]
    Ω_aibj[a, i, b, j] -= F_oo[k,j] * u[a,i,b,k]
    Ω_aibj[a, i, b, j] -= t1s[a,k] * L_vooo[b,j,k,i]
    Ω_aibj[a, i, b, j] -= t1s[b,k] * L_vooo[a,i,k,j]
    Ω_aibj[a, i, b, j] += t1s[c,i] * L_vvvo[a,c,b,j]
    Ω_aibj[a, i, b, j] += t1s[c,j] * L_vovv[a,i,b,c]
    Ω_aibj[a, i, b, j] -= L_voov[a,j,k,c] * t2s[b,i,c,k]
    Ω_aibj[a, i, b, j] += L_vvvv[a,c,b,d] * t2s[c,i,d,j]
    Ω_aibj[a, i, b, j] -= L_vvoo[a,c,k,j] * t2s[b,k,c,i]
    Ω_aibj[a, i, b, j] -= L_voov[b,i,k,c] * t2s[a,j,c,k]
    Ω_aibj[a, i, b, j] -= L_vvoo[b,c,k,i] * t2s[a,k,c,j]
    Ω_aibj[a, i, b, j] += L_oooo[k,i,l,j] * t2s[a,k,b,l]
    Ω_aibj[a, i, b, j] += L_voov[a,i,k,c] * u[b,j,c,k]
    Ω_aibj[a, i, b, j] += L_voov[b,j,k,c] * u[a,i,c,k]
    Ω_aibj[a, i, b, j] += γv[a] * γo[k] * L_vooo[b,j,k,i]
    Ω_aibj[a, i, b, j] += γo[i] * γv[c] * L_vvvo[a,c,b,j]
    Ω_aibj[a, i, b, j] += γv[b] * γo[k] * L_vooo[a,i,k,j]
    Ω_aibj[a, i, b, j] += γo[j] * γv[c] * L_vovv[a,i,b,c]
    Ω_aibj[a, i, b, j] -= F_ov[k,c] * t1s[a,k] * u[b,j,c,i]
    Ω_aibj[a, i, b, j] -= F_ov[k,c] * t1s[b,k] * u[a,i,c,j]
    Ω_aibj[a, i, b, j] -= F_ov[k,c] * t1s[c,i] * u[a,k,b,j]
    Ω_aibj[a, i, b, j] -= F_ov[k,c] * t1s[c,j] * u[a,i,b,k]
    Ω_aibj[a, i, b, j] += t1s[a,k] * t1s[b,l] * L_oooo[k,i,l,j]
    Ω_aibj[a, i, b, j] -= t1s[a,k] * t1s[c,i] * L_voov[b,j,k,c]
    Ω_aibj[a, i, b, j] -= t1s[a,k] * t1s[c,j] * L_vvoo[b,c,k,i]
    Ω_aibj[a, i, b, j] -= t1s[b,k] * t1s[c,i] * L_vvoo[a,c,k,j]
    Ω_aibj[a, i, b, j] -= t1s[b,k] * t1s[c,j] * L_voov[a,i,k,c]
    Ω_aibj[a, i, b, j] += t1s[c,i] * t1s[d,j] * L_vvvv[a,c,b,d]
    Ω_aibj[a, i, b, j] -= t1s[a,k] * L_vvov[b,c,k,d] * t2s[c,j,d,i]
    Ω_aibj[a, i, b, j] += t1s[a,k] * L_ooov[k,j,l,c] * t2s[b,i,c,l]
    Ω_aibj[a, i, b, j] += t1s[a,k] * L_ovoo[k,c,l,j] * t2s[b,l,c,i]
    Ω_aibj[a, i, b, j] -= t1s[b,k] * L_vvov[a,c,k,d] * t2s[c,i,d,j]
    Ω_aibj[a, i, b, j] += t1s[b,k] * L_ooov[k,i,l,c] * t2s[a,j,c,l]
    Ω_aibj[a, i, b, j] += t1s[b,k] * L_ovoo[k,c,l,i] * t2s[a,l,c,j]
    Ω_aibj[a, i, b, j] -= t1s[c,i] * L_vvov[b,c,k,d] * t2s[a,j,d,k]
    Ω_aibj[a, i, b, j] -= t1s[c,i] * L_vvov[b,d,k,c] * t2s[a,k,d,j]
    Ω_aibj[a, i, b, j] += t1s[c,i] * L_ooov[k,j,l,c] * t2s[a,l,b,k]
    Ω_aibj[a, i, b, j] -= t1s[c,j] * L_vvov[a,c,k,d] * t2s[b,i,d,k]
    Ω_aibj[a, i, b, j] -= t1s[c,j] * L_vvov[a,d,k,c] * t2s[b,k,d,i]
    Ω_aibj[a, i, b, j] += t1s[c,j] * L_ooov[k,i,l,c] * t2s[a,k,b,l]
    Ω_aibj[a, i, b, j] -= t1s[a,k] * L_ooov[k,i,l,c] * u[b,j,c,l]
    Ω_aibj[a, i, b, j] -= t1s[b,k] * L_ooov[k,j,l,c] * u[a,i,c,l]
    Ω_aibj[a, i, b, j] += t1s[c,i] * L_vvov[a,c,k,d] * u[b,j,d,k]
    Ω_aibj[a, i, b, j] += t1s[c,j] * L_vvov[b,c,k,d] * u[a,i,d,k]
    Ω_aibj[a, i, b, j] += t1s[c,k] * L_vvov[a,d,k,c] * u[b,j,d,i]
    Ω_aibj[a, i, b, j] += t1s[c,k] * L_vvov[b,d,k,c] * u[a,i,d,j]
    Ω_aibj[a, i, b, j] -= t1s[c,k] * L_ovoo[k,c,l,i] * u[a,l,b,j]
    Ω_aibj[a, i, b, j] -= t1s[c,k] * L_ovoo[k,c,l,j] * u[a,i,b,l]
    Ω_aibj[a, i, b, j] += L_ovov[k,c,l,d] * t2s[a,j,c,k] * t2s[b,l,d,i]
    Ω_aibj[a, i, b, j] += L_ovov[k,c,l,d] * t2s[a,k,b,l] * t2s[c,i,d,j]
    Ω_aibj[a, i, b, j] += L_ovov[k,c,l,d] * t2s[a,k,d,j] * t2s[b,l,c,i]
    Ω_aibj[a, i, b, j] -= L_ovov[k,c,l,d] * t2s[a,k,d,l] * u[b,j,c,i]
    Ω_aibj[a, i, b, j] -= L_ovov[k,c,l,d] * t2s[b,i,c,k] * u[a,j,d,l]
    Ω_aibj[a, i, b, j] -= L_ovov[k,c,l,d] * t2s[b,k,d,l] * u[a,i,c,j]
    Ω_aibj[a, i, b, j] -= L_ovov[k,c,l,d] * t2s[c,i,d,l] * u[a,k,b,j]
    Ω_aibj[a, i, b, j] -= L_ovov[k,c,l,d] * t2s[c,j,d,l] * u[a,i,b,k]
    Ω_aibj[a, i, b, j] += L_ovov[k,c,l,d] * u[a,i,c,k] * u[b,j,d,l]
    Ω_aibj[a, i, b, j] += γv[a] * γo[k] * F_ov[k,c] * u[b,j,c,i]
    Ω_aibj[a, i, b, j] -= γo[i] * γv[c] * F_ov[k,c] * u[a,k,b,j]
    Ω_aibj[a, i, b, j] += γv[b] * γo[k] * F_ov[k,c] * u[a,i,c,j]
    Ω_aibj[a, i, b, j] -= γo[j] * γv[c] * F_ov[k,c] * u[a,i,b,k]
    Ω_aibj[a, i, b, j] -= γv[a] * γo[k] * t1s[b,l] * L_oooo[k,i,l,j]
    Ω_aibj[a, i, b, j] += γv[a] * γo[k] * t1s[c,i] * L_voov[b,j,k,c]
    Ω_aibj[a, i, b, j] += γv[a] * γo[k] * t1s[c,j] * L_vvoo[b,c,k,i]
    Ω_aibj[a, i, b, j] -= γo[i] * γv[c] * t1s[a,k] * L_voov[b,j,k,c]
    Ω_aibj[a, i, b, j] -= γo[i] * γv[c] * t1s[b,k] * L_vvoo[a,c,k,j]
    Ω_aibj[a, i, b, j] += γo[i] * γv[c] * t1s[d,j] * L_vvvv[a,c,b,d]
    Ω_aibj[a, i, b, j] -= γv[b] * γo[k] * t1s[a,l] * L_oooo[k,j,l,i]
    Ω_aibj[a, i, b, j] += γv[b] * γo[k] * t1s[c,i] * L_vvoo[a,c,k,j]
    Ω_aibj[a, i, b, j] += γv[b] * γo[k] * t1s[c,j] * L_voov[a,i,k,c]
    Ω_aibj[a, i, b, j] -= γo[j] * γv[c] * t1s[a,k] * L_vvoo[b,c,k,i]
    Ω_aibj[a, i, b, j] -= γo[j] * γv[c] * t1s[b,k] * L_voov[a,i,k,c]
    Ω_aibj[a, i, b, j] += γo[j] * γv[c] * t1s[d,i] * L_vvvv[a,d,b,c]
    Ω_aibj[a, i, b, j] += γv[a] * γo[k] * L_vvov[b,c,k,d] * t2s[c,j,d,i]
    Ω_aibj[a, i, b, j] -= γv[a] * γo[k] * L_ooov[k,j,l,c] * t2s[b,i,c,l]
    Ω_aibj[a, i, b, j] -= γv[a] * γo[k] * L_ovoo[k,c,l,j] * t2s[b,l,c,i]
    Ω_aibj[a, i, b, j] -= γo[i] * γv[c] * L_vvov[b,c,k,d] * t2s[a,j,d,k]
    Ω_aibj[a, i, b, j] -= γo[i] * γv[c] * L_vvov[b,d,k,c] * t2s[a,k,d,j]
    Ω_aibj[a, i, b, j] += γo[i] * γv[c] * L_ooov[k,j,l,c] * t2s[a,l,b,k]
    Ω_aibj[a, i, b, j] += γv[b] * γo[k] * L_vvov[a,c,k,d] * t2s[c,i,d,j]
    Ω_aibj[a, i, b, j] -= γv[b] * γo[k] * L_ooov[k,i,l,c] * t2s[a,j,c,l]
    Ω_aibj[a, i, b, j] -= γv[b] * γo[k] * L_ovoo[k,c,l,i] * t2s[a,l,c,j]
    Ω_aibj[a, i, b, j] -= γo[j] * γv[c] * L_vvov[a,c,k,d] * t2s[b,i,d,k]
    Ω_aibj[a, i, b, j] -= γo[j] * γv[c] * L_vvov[a,d,k,c] * t2s[b,k,d,i]
    Ω_aibj[a, i, b, j] += γo[j] * γv[c] * L_ooov[k,i,l,c] * t2s[a,k,b,l]
    Ω_aibj[a, i, b, j] += γv[a] * γo[k] * L_ooov[k,i,l,c] * u[b,j,c,l]
    Ω_aibj[a, i, b, j] += γo[i] * γv[c] * L_vvov[a,c,k,d] * u[b,j,d,k]
    Ω_aibj[a, i, b, j] += γv[b] * γo[k] * L_ooov[k,j,l,c] * u[a,i,c,l]
    Ω_aibj[a, i, b, j] += γo[j] * γv[c] * L_vvov[b,c,k,d] * u[a,i,d,k]
    Ω_aibj[a, i, b, j] += t1s[a,k] * t1s[b,l] * t1s[c,i] * L_ovoo[k,c,l,j]
    Ω_aibj[a, i, b, j] += t1s[a,k] * t1s[b,l] * t1s[c,j] * L_ooov[k,i,l,c]
    Ω_aibj[a, i, b, j] -= t1s[a,k] * t1s[c,i] * t1s[d,j] * L_vvov[b,d,k,c]
    Ω_aibj[a, i, b, j] -= t1s[b,k] * t1s[c,i] * t1s[d,j] * L_vvov[a,c,k,d]
    Ω_aibj[a, i, b, j] += t1s[a,k] * t1s[b,l] * L_ovov[k,c,l,d] * t2s[c,i,d,j]
    Ω_aibj[a, i, b, j] += t1s[a,k] * t1s[c,j] * L_ovov[k,c,l,d] * t2s[b,i,d,l]
    Ω_aibj[a, i, b, j] += t1s[a,k] * t1s[c,j] * L_ovov[k,d,l,c] * t2s[b,l,d,i]
    Ω_aibj[a, i, b, j] += t1s[b,k] * t1s[c,i] * L_ovov[k,c,l,d] * t2s[a,j,d,l]
    Ω_aibj[a, i, b, j] += t1s[b,k] * t1s[c,i] * L_ovov[k,d,l,c] * t2s[a,l,d,j]
    Ω_aibj[a, i, b, j] += t1s[c,i] * t1s[d,j] * L_ovov[k,c,l,d] * t2s[a,k,b,l]
    Ω_aibj[a, i, b, j] -= t1s[a,k] * t1s[c,i] * L_ovov[k,c,l,d] * u[b,j,d,l]
    Ω_aibj[a, i, b, j] -= t1s[a,k] * t1s[c,l] * L_ovov[k,d,l,c] * u[b,j,d,i]
    Ω_aibj[a, i, b, j] -= t1s[b,k] * t1s[c,j] * L_ovov[k,c,l,d] * u[a,i,d,l]
    Ω_aibj[a, i, b, j] -= t1s[b,k] * t1s[c,l] * L_ovov[k,d,l,c] * u[a,i,d,j]
    Ω_aibj[a, i, b, j] -= t1s[c,i] * t1s[d,k] * L_ovov[k,d,l,c] * u[a,l,b,j]
    Ω_aibj[a, i, b, j] -= t1s[c,j] * t1s[d,k] * L_ovov[k,d,l,c] * u[a,i,b,l]
    Ω_aibj[a, i, b, j] -= γv[a] * γo[k] * t1s[b,l] * t1s[c,i] * L_ovoo[k,c,l,j]
    Ω_aibj[a, i, b, j] -= γv[a] * γo[k] * t1s[b,l] * t1s[c,j] * L_ooov[k,i,l,c]
    Ω_aibj[a, i, b, j] += γv[a] * γo[k] * t1s[c,i] * t1s[d,j] * L_vvov[b,d,k,c]
    Ω_aibj[a, i, b, j] += γo[i] * γv[c] * t1s[a,k] * t1s[b,l] * L_ovoo[k,c,l,j]
    Ω_aibj[a, i, b, j] -= γo[i] * γv[c] * t1s[a,k] * t1s[d,j] * L_vvov[b,d,k,c]
    Ω_aibj[a, i, b, j] -= γo[i] * γv[c] * t1s[b,k] * t1s[d,j] * L_vvov[a,c,k,d]
    Ω_aibj[a, i, b, j] -= γv[b] * γo[k] * t1s[a,l] * t1s[c,i] * L_ooov[k,j,l,c]
    Ω_aibj[a, i, b, j] -= γv[b] * γo[k] * t1s[a,l] * t1s[c,j] * L_ovoo[k,c,l,i]
    Ω_aibj[a, i, b, j] += γv[b] * γo[k] * t1s[c,i] * t1s[d,j] * L_vvov[a,c,k,d]
    Ω_aibj[a, i, b, j] += γo[j] * γv[c] * t1s[a,k] * t1s[b,l] * L_ooov[k,i,l,c]
    Ω_aibj[a, i, b, j] -= γo[j] * γv[c] * t1s[a,k] * t1s[d,i] * L_vvov[b,c,k,d]
    Ω_aibj[a, i, b, j] -= γo[j] * γv[c] * t1s[b,k] * t1s[d,i] * L_vvov[a,d,k,c]
    Ω_aibj[a, i, b, j] -= γv[a] * γo[k] * t1s[b,l] * L_ovov[k,c,l,d] * t2s[c,i,d,j]
    Ω_aibj[a, i, b, j] -= γv[a] * γo[k] * t1s[c,j] * L_ovov[k,c,l,d] * t2s[b,i,d,l]
    Ω_aibj[a, i, b, j] -= γv[a] * γo[k] * t1s[c,j] * L_ovov[k,d,l,c] * t2s[b,l,d,i]
    Ω_aibj[a, i, b, j] += γo[i] * γv[c] * t1s[b,k] * L_ovov[k,c,l,d] * t2s[a,j,d,l]
    Ω_aibj[a, i, b, j] += γo[i] * γv[c] * t1s[b,k] * L_ovov[k,d,l,c] * t2s[a,l,d,j]
    Ω_aibj[a, i, b, j] += γo[i] * γv[c] * t1s[d,j] * L_ovov[k,c,l,d] * t2s[a,k,b,l]
    Ω_aibj[a, i, b, j] -= γv[b] * γo[k] * t1s[a,l] * L_ovov[k,c,l,d] * t2s[c,j,d,i]
    Ω_aibj[a, i, b, j] -= γv[b] * γo[k] * t1s[c,i] * L_ovov[k,c,l,d] * t2s[a,j,d,l]
    Ω_aibj[a, i, b, j] -= γv[b] * γo[k] * t1s[c,i] * L_ovov[k,d,l,c] * t2s[a,l,d,j]
    Ω_aibj[a, i, b, j] += γo[j] * γv[c] * t1s[a,k] * L_ovov[k,c,l,d] * t2s[b,i,d,l]
    Ω_aibj[a, i, b, j] += γo[j] * γv[c] * t1s[a,k] * L_ovov[k,d,l,c] * t2s[b,l,d,i]
    Ω_aibj[a, i, b, j] += γo[j] * γv[c] * t1s[d,i] * L_ovov[k,c,l,d] * t2s[a,l,b,k]
    Ω_aibj[a, i, b, j] += γv[a] * γo[k] * t1s[c,i] * L_ovov[k,c,l,d] * u[b,j,d,l]
    Ω_aibj[a, i, b, j] += γv[a] * γo[k] * t1s[c,l] * L_ovov[k,d,l,c] * u[b,j,d,i]
    Ω_aibj[a, i, b, j] -= γo[i] * γv[c] * t1s[a,k] * L_ovov[k,c,l,d] * u[b,j,d,l]
    Ω_aibj[a, i, b, j] -= γo[i] * γv[c] * t1s[d,k] * L_ovov[k,d,l,c] * u[a,l,b,j]
    Ω_aibj[a, i, b, j] += γv[b] * γo[k] * t1s[c,j] * L_ovov[k,c,l,d] * u[a,i,d,l]
    Ω_aibj[a, i, b, j] += γv[b] * γo[k] * t1s[c,l] * L_ovov[k,d,l,c] * u[a,i,d,j]
    Ω_aibj[a, i, b, j] -= γo[j] * γv[c] * t1s[b,k] * L_ovov[k,c,l,d] * u[a,i,d,l]
    Ω_aibj[a, i, b, j] -= γo[j] * γv[c] * t1s[d,k] * L_ovov[k,d,l,c] * u[a,i,b,l]
    Ω_aibj[a, i, b, j] += t1s[a,k] * t1s[b,l] * t1s[c,i] * t1s[d,j] * L_ovov[k,c,l,d]
    Ω_aibj[a, i, b, j] -= γv[a] * γo[k] * t1s[b,l] * t1s[c,i] * t1s[d,j] * L_ovov[k,c,l,d]
    Ω_aibj[a, i, b, j] += γo[i] * γv[c] * t1s[a,k] * t1s[b,l] * t1s[d,j] * L_ovov[k,c,l,d]
    Ω_aibj[a, i, b, j] -= γv[b] * γo[k] * t1s[a,l] * t1s[c,i] * t1s[d,j] * L_ovov[k,d,l,c]
    Ω_aibj[a, i, b, j] += γo[j] * γv[c] * t1s[a,k] * t1s[b,l] * t1s[d,i] * L_ovov[k,d,l,c]
   end
   #symmetric_sum!(no, nv, Ω_aibj)
end

function symmetric_sum!(no, nv, aibj)
	Ω2 = copy(aibj)
	@tensoropt aibj[a, i, b, j] += Ω2[b, j, a, i]
	aibj.*=0.5
end

function eta_func_i!(no, nv, γo, γv, t1s, t2s, λ, λdag, eta_i)
	λo = @view λ[1:no]
	λv = @view λ[(no+1):(no+nv)]
	λdago = @view λdag[1:no]
	λdagv = @view λdag[(no+1):(no+nv)]
	@tensoropt eta_i[i] = -2 * λdago[i]
end

function eta_func_a!(no, nv, γo, γv, t1s, t2s, λ, λdag, eta_a)
	λo = @view λ[1:no]
	λv = @view λ[(no+1):(no+nv)]
	λdago = @view λdag[1:no]
	λdagv = @view λdag[(no+1):(no+nv)]
	@tensoropt eta_a[a] = 2 * λv[a]
end

function eta_func_ai!(no, nv, γo, γv, t1s, t2s, λ, λdag, eta_ai)
	λo = @view λ[1:no]
	λv = @view λ[(no+1):(no+nv)]
	λdago = @view λdag[1:no]
	λdagv = @view λdag[(no+1):(no+nv)]
	@tensoropt begin
		eta_ai[a, i] = 2 * F_ov[i,a]
		eta_ai[a, i] += 2 * L_ovov[i,a,j,b] * t1s[b,j]
	end
end

function eta_func_aibj!(no, nv, γo, γv, t1s, t2s, λ, λdag, eta_aibj)
	λo = @view λ[1:no]
	λv = @view λ[(no+1):(no+nv)]
	λdago = @view λdag[1:no]
	λdagv = @view λdag[(no+1):(no+nv)]
	@tensoropt eta_aibj[a, i, b, j] = L_ovov[i, a, j, b]
end

