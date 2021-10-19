
abstract type LossFunction end

struct HuberLoss <: LossFunction
    δ::Float64
    c_ptr::Ptr{Cvoid}
end

function HuberLoss(δ::Float64)
    c_ptr = ceres_create_huber_loss_c(δ)
    HuberLoss(δ, c_ptr)
end

struct EmptyLoss <: LossFunction end
