function [img_pat_loc] = get_img_pat_loc(pat)

        crop_height = 96;
        img_pat_loc = zeros(crop_height*crop_height, crop_height*crop_height);
        iter = 1;
        for j = 1:crop_height
                for i = 1:crop_height

                        img_pat = zeros(crop_height, crop_height);
                        img_pat(i,j) = 1;

                        st_pos_i = min(max(i - pat, 1), crop_height);
                        st_pos_j = min(max(j - pat, 1), crop_height);
                        end_pos_i = min(max(st_pos_i+2*pat,1), crop_height);
                        end_pos_j = min(max(st_pos_j+2*pat,1), crop_height);

                        img_pat(st_pos_i:end_pos_i, st_pos_j:end_pos_j) = 1;
                        img_pat_loc(iter,:) = img_pat(:);
                        iter = iter+1;
                end
        end
end
