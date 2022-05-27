#!/bin/bash

gpg --import ../gpg-keys/generator.gpg #Generator key (public)
gpg --pinentry-mode=loopback --passphrase-file ../gpg-keys/gpg-passphrase.txt --import ../gpg-keys/device-secret-key.gpg #Device private and public keys


### VERIFY LICENCE (on the device!!!!!)###
gpg --output license_decrypted.txt --pinentry-mode=loopback --passphrase-file ../gpg-keys/gpg-passphrase.txt --decrypt ../license/license.txt

decrypted_signature=$(gpg --decrypt <<EOF
$(sed -n -e '/-----BEGIN PGP MESSAGE-----/,/-----END PGP MESSAGE-----/p' license_decrypted.txt)
EOF
)

decrypted_signature=$(echo ${decrypted_signature} | head -1) #Useless ??

license=$(sed -e '1,/-----END PGP MESSAGE-----/ d' license_decrypted.txt)

recomputed_signature=$(shasum -a 256 <<EOF
$license
EOF
)

echo $1
device_identifier=$1 

license_device_identifier=$(echo $license | jq .device_identifier)
license_device_identifier=${license_device_identifier//$'"'/} #Remove "
echo $license_device_identifier

echo $2
model=$2
model_licensed=$(echo $license | jq .models_licensed.$model)
start_date=$(echo $license | jq .start_date)
end_date=$(echo $license | jq .end_date)

recomputed_signature=$(echo $recomputed_signature | awk '{print $1}')

if [[ "$recomputed_signature" == "$decrypted_signature" ]]; then
    echo "it's okay"
    #Continue
    
    if [[ "$device_identifier" == "$license_device_identifier" ]]; then
        echo "License for this device";
        if [[ "$model_licensed" == "true" ]]; then
        echo "license ok";
        else
            echo "license not ok";
        fi

        echo $start_date > ../license/validation.txt;
        echo $end_date >> ../license/validation.txt;

    else
        echo "License not generated for this device";
        echo "Wrong device" > ../license/validation.txt;
    fi

else
    echo "Wrong license signature";
    echo "INVALID" > ../license/validation.txt;
fi




